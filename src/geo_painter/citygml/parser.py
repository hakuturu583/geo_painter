"""CityGML パーサー実装

CityGMLScanner: ZIPアーカイブを走査してGMLファイルを列挙する
CityGMLParser:  lxmlを使ってGMLファイルを解析し Geometry リストを返す
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import IO, Iterator

import numpy as np
from lxml import etree

from geo_painter.citygml.models import FeatureType, GmlPolygon, Geometry, Ring

logger = logging.getLogger(__name__)

# PLATEAU CityGML 2.0 名前空間
NS: dict[str, str] = {
    "gml": "http://www.opengis.net/gml",
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "tran": "http://www.opengis.net/citygml/transportation/2.0",
    "dem": "http://www.opengis.net/citygml/relief/2.0",
}

# 地物種別ごとのLODフォールバックXPath定義
# タプル: (lod_value, xpath_relative_to_cityobject_member)
_BUILDING_LOD_XPATHS: list[tuple[int, str]] = [
    (3, ".//bldg:lod3Solid"),
    (2, ".//bldg:lod2Solid"),
    (2, ".//bldg:lod2MultiSurface"),
    (1, ".//bldg:lod1Solid"),
    (1, ".//bldg:lod1MultiSurface"),
    (0, ".//bldg:lod0FootPrint"),
]

_ROAD_LOD_XPATHS: list[tuple[int, str]] = [
    (2, ".//tran:lod2MultiSurface"),
    (1, ".//tran:lod1MultiSurface"),
]

# 地形はTINRelief内のTriangleを直接抽出
_TERRAIN_XPATH = ".//gml:Triangle"


class CityGMLScanner:
    """ZIPアーカイブ内のGMLファイルを地物種別ごとに列挙するクラス

    Args:
        zip_path: ZIPファイルのパス
    """

    def __init__(self, zip_path: Path) -> None:
        self._zip_path = zip_path

    def iter_files(self) -> Iterator[tuple[FeatureType, IO[bytes]]]:
        """ZIPアーカイブ内の udx/{bldg,tran,dem}/*.gml を順に返す

        Yields:
            (feature_type, file_stream) のタプル
        """
        with zipfile.ZipFile(self._zip_path, "r") as zf:
            for name in zf.namelist():
                feature_type = self._classify(name)
                if feature_type is None:
                    continue
                if not name.lower().endswith(".gml"):
                    continue
                logger.debug("GML ファイルを処理: %s", name)
                with zf.open(name) as f:
                    yield feature_type, f

    @staticmethod
    def _classify(zip_entry: str) -> FeatureType | None:
        """ZIPエントリのパスから地物種別を判定する

        udx/<subdir>/*.gml のサブディレクトリ名を基準に判定し、
        マッチしない場合はファイル名のプレフィックスでフォールバックする。
        """
        parts = zip_entry.replace("\\", "/").split("/")
        # udx/<subdir>/ パターンを探す
        for i, part in enumerate(parts):
            if part == "udx" and i + 1 < len(parts):
                subdir = parts[i + 1]
                for ft in FeatureType:
                    if subdir == ft.udx_dir:
                        return ft
                break

        # フォールバック: ファイル名から判定
        fname = parts[-1].lower() if parts else ""
        if fname.startswith("bldg_"):
            return FeatureType.BUILDING
        if fname.startswith("tran_"):
            return FeatureType.ROAD
        if fname.startswith("dem_"):
            return FeatureType.TERRAIN

        return None


class CityGMLParser:
    """lxmlを使ってCityGMLを解析し Geometry リストを返すクラス"""

    def parse(self, stream: IO[bytes], feature_type: FeatureType) -> list[Geometry]:
        """GMLストリームを解析して Geometry リストを返す

        Args:
            stream: GMLファイルのバイトストリーム
            feature_type: 地物種別

        Returns:
            Geometry のリスト
        """
        try:
            tree = etree.parse(stream)
        except etree.XMLSyntaxError as exc:
            logger.warning("XML パース失敗: %s", exc)
            return []

        root = tree.getroot()
        geometries: list[Geometry] = []

        if feature_type == FeatureType.TERRAIN:
            geometries.extend(self._parse_terrain(root))
        else:
            lod_xpaths = (
                _BUILDING_LOD_XPATHS
                if feature_type == FeatureType.BUILDING
                else _ROAD_LOD_XPATHS
            )
            members = root.findall(".//core:cityObjectMember", NS)
            if not members:
                # core: 名前空間なしのフォールバック
                members = root.findall(".//{http://www.opengis.net/citygml/2.0}cityObjectMember")
            for member in members:
                geom = self._parse_member(member, feature_type, lod_xpaths)
                if geom is not None:
                    geometries.append(geom)

        return geometries

    # ------------------------------------------------------------------
    # 建物・道路の解析
    # ------------------------------------------------------------------

    def _parse_member(
        self,
        member: etree._Element,
        feature_type: FeatureType,
        lod_xpaths: list[tuple[int, str]],
    ) -> Geometry | None:
        """cityObjectMember 要素から Geometry を生成する"""
        # gml:id 取得
        gml_id_attr = "{http://www.opengis.net/gml}id"
        feature_id = member.get(gml_id_attr, "")
        if not feature_id:
            # 子要素から id を探す
            children = list(member)
            if children:
                feature_id = children[0].get(gml_id_attr, "unknown")

        # LODフォールバック: 上位のXPathから順に試す
        for lod, xpath in lod_xpaths:
            surfaces = member.findall(xpath, NS)
            if not surfaces:
                continue
            polygons = self._extract_polygons(surfaces)
            if polygons:
                return Geometry(
                    feature_type=feature_type,
                    feature_id=feature_id,
                    lod=lod,
                    polygons=polygons,
                )

        return None

    def _extract_polygons(self, surface_elements: list[etree._Element]) -> list[GmlPolygon]:
        """サーフェス要素群からGmlPolygonリストを抽出する"""
        polygons: list[GmlPolygon] = []
        for surf in surface_elements:
            for poly_elem in surf.iter("{http://www.opengis.net/gml}Polygon"):
                poly = self._parse_polygon(poly_elem)
                if poly is not None:
                    polygons.append(poly)
        return polygons

    def _parse_polygon(self, poly_elem: etree._Element) -> GmlPolygon | None:
        """gml:Polygon 要素から GmlPolygon を生成する"""
        # exterior
        ext_elem = poly_elem.find("gml:exterior/gml:LinearRing", NS)
        if ext_elem is None:
            return None
        exterior = self._parse_ring(ext_elem)
        if exterior is None or len(exterior.coords) < 3:
            return None

        # interior（穴）
        interiors: list[Ring] = []
        for int_ring in poly_elem.findall("gml:interior/gml:LinearRing", NS):
            ring = self._parse_ring(int_ring)
            if ring is not None and len(ring.coords) >= 3:
                interiors.append(ring)

        return GmlPolygon(exterior=exterior, interiors=interiors)

    def _parse_ring(self, ring_elem: etree._Element) -> Ring | None:
        """gml:LinearRing 要素から Ring を生成する

        gml:posList を優先し、なければ gml:pos を結合する。
        末尾の重複点（閉合点）は除去する。
        """
        # gml:posList
        pos_list_elem = ring_elem.find("gml:posList", NS)
        if pos_list_elem is not None and pos_list_elem.text:
            coords = self._parse_pos_list(pos_list_elem.text)
        else:
            # gml:pos フォールバック
            pos_elems = ring_elem.findall("gml:pos", NS)
            if not pos_elems:
                return None
            values: list[float] = []
            for pos in pos_elems:
                if pos.text:
                    values.extend(float(v) for v in pos.text.split())
            if not values:
                return None
            coords = np.array(values, dtype=np.float64).reshape(-1, 3)

        if coords is None or len(coords) < 3:
            return None

        # 末尾の重複点を除去（閉合: 最初と最後が同じ座標）
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        if len(coords) < 3:
            return None

        return Ring(coords=coords)

    @staticmethod
    def _parse_pos_list(text: str) -> np.ndarray | None:
        """posList テキストを (N, 3) の numpy 配列に変換する"""
        try:
            values = [float(v) for v in text.split()]
        except ValueError:
            return None
        if len(values) % 3 != 0:
            return None
        return np.array(values, dtype=np.float64).reshape(-1, 3)

    # ------------------------------------------------------------------
    # 地形の解析
    # ------------------------------------------------------------------

    def _parse_terrain(self, root: etree._Element) -> list[Geometry]:
        """TINRelief内のgml:Triangleを解析してGeometryリストを返す"""
        geometries: list[Geometry] = []

        for triangle_elem in root.iter("{http://www.opengis.net/gml}Triangle"):
            ring_elem = triangle_elem.find("gml:exterior/gml:LinearRing", NS)
            if ring_elem is None:
                continue
            ring = self._parse_ring(ring_elem)
            if ring is None or len(ring.coords) < 3:
                continue
            poly = GmlPolygon(exterior=ring)
            geom = Geometry(
                feature_type=FeatureType.TERRAIN,
                feature_id="terrain",
                lod=0,
                polygons=[poly],
            )
            geometries.append(geom)

        return geometries
