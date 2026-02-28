"""CityGML パーサー実装

CityGMLScanner: ZIPアーカイブを走査してGMLファイルを列挙する
CityGMLParser:  lxmlを使ってGMLファイルを解析し Geometry リストを返す
"""

from __future__ import annotations

import io
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
    "app": "http://www.opengis.net/citygml/appearance/2.0",
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

_GML_ID = "{http://www.opengis.net/gml}id"
_APP_NS = "http://www.opengis.net/citygml/appearance/2.0"


class CityGMLScanner:
    """ZIPアーカイブ内のGMLファイルを地物種別ごとに列挙するクラス

    Args:
        zip_path: ZIPファイルのパス
    """

    def __init__(self, zip_path: Path) -> None:
        self._zip_path = zip_path

    def iter_files(self) -> Iterator[tuple[FeatureType, IO[bytes], str]]:
        """ZIPアーカイブ内の udx/{bldg,tran,dem}/*.gml を順に返す

        Yields:
            (feature_type, file_stream, zip_entry_name) のタプル
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
                    data = io.BytesIO(f.read())
                yield feature_type, data, name

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

        # app:Appearance をパースしてリングID→(URI, UV) マップを構築
        tex_map = self._parse_appearances(root)

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
                geom = self._parse_member(member, feature_type, lod_xpaths, tex_map)
                if geom is not None:
                    geometries.append(geom)

        return geometries

    # ------------------------------------------------------------------
    # Appearance（テクスチャ）の解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_appearances(root: etree._Element) -> dict[str, tuple[str, np.ndarray]]:
        """app:ParameterizedTexture から ring_id → (image_uri, uv_array) マップを構築する

        Returns:
            ring の gml:id をキーに (imageURI, UV座標配列 shape=(N,2)) を値とする辞書
        """
        result: dict[str, tuple[str, np.ndarray]] = {}

        for param_tex in root.iter(f"{{{_APP_NS}}}ParameterizedTexture"):
            uri_elem = param_tex.find("app:imageURI", NS)
            if uri_elem is None or not uri_elem.text:
                continue
            image_uri = uri_elem.text.strip()

            for target in param_tex.findall("app:target", NS):
                tex_coord_list = target.find("app:TexCoordList", NS)
                if tex_coord_list is None:
                    continue
                for tex_coords in tex_coord_list.findall("app:textureCoordinates", NS):
                    ring_ref = tex_coords.get("ring", "")  # e.g. "#UUID_exterior"
                    ring_id = ring_ref.lstrip("#")
                    if ring_id and tex_coords.text:
                        try:
                            vals = [float(v) for v in tex_coords.text.split()]
                            if len(vals) % 2 == 0 and vals:
                                uv = np.array(vals, dtype=np.float32).reshape(-1, 2)
                                result[ring_id] = (image_uri, uv)
                        except ValueError:
                            pass

        return result

    # ------------------------------------------------------------------
    # 建物・道路の解析
    # ------------------------------------------------------------------

    def _parse_member(
        self,
        member: etree._Element,
        feature_type: FeatureType,
        lod_xpaths: list[tuple[int, str]],
        tex_map: dict[str, tuple[str, np.ndarray]] | None = None,
    ) -> Geometry | None:
        """cityObjectMember 要素から Geometry を生成する"""
        # gml:id 取得
        feature_id = member.get(_GML_ID, "")
        if not feature_id:
            # 子要素から id を探す
            children = list(member)
            if children:
                feature_id = children[0].get(_GML_ID, "unknown")

        # LODフォールバック: 上位のXPathから順に試す
        for lod, xpath in lod_xpaths:
            surfaces = member.findall(xpath, NS)
            if not surfaces:
                continue
            polygons = self._extract_polygons(surfaces, tex_map)
            if polygons:
                return Geometry(
                    feature_type=feature_type,
                    feature_id=feature_id,
                    lod=lod,
                    polygons=polygons,
                )

        return None

    def _extract_polygons(
        self,
        surface_elements: list[etree._Element],
        tex_map: dict[str, tuple[str, np.ndarray]] | None = None,
    ) -> list[GmlPolygon]:
        """サーフェス要素群からGmlPolygonリストを抽出する"""
        polygons: list[GmlPolygon] = []
        for surf in surface_elements:
            for poly_elem in surf.iter("{http://www.opengis.net/gml}Polygon"):
                poly = self._parse_polygon(poly_elem, tex_map)
                if poly is not None:
                    polygons.append(poly)
        return polygons

    def _parse_polygon(
        self,
        poly_elem: etree._Element,
        tex_map: dict[str, tuple[str, np.ndarray]] | None = None,
    ) -> GmlPolygon | None:
        """gml:Polygon 要素から GmlPolygon を生成する"""
        # exterior
        ext_elem = poly_elem.find("gml:exterior/gml:LinearRing", NS)
        if ext_elem is None:
            return None
        exterior = self._parse_ring(ext_elem)
        if exterior is None or len(exterior.coords) < 3:
            return None

        # interior（穴）
        int_elems = poly_elem.findall("gml:interior/gml:LinearRing", NS)
        interiors: list[Ring] = []
        for int_elem in int_elems:
            ring = self._parse_ring(int_elem)
            if ring is not None and len(ring.coords) >= 3:
                interiors.append(ring)

        poly = GmlPolygon(exterior=exterior, interiors=interiors)

        # テクスチャUV の注入
        if tex_map:
            self._inject_texture(poly, ext_elem, int_elems, tex_map)

        return poly

    @staticmethod
    def _inject_texture(
        poly: GmlPolygon,
        ext_elem: etree._Element,
        int_elems: list[etree._Element],
        tex_map: dict[str, tuple[str, np.ndarray]],
    ) -> None:
        """GmlPolygon の各リングにテクスチャUVを注入する"""
        ext_id = ext_elem.get(_GML_ID, "")
        if ext_id in tex_map:
            img_uri, uv = tex_map[ext_id]
            n = len(poly.exterior.coords)
            # textureCoordinates は閉合点を含む場合があるので trim
            if len(uv) >= n:
                poly.exterior.uv = uv[:n]
                poly.texture_uri = img_uri
            else:
                logger.debug(
                    "UV数(%d)と頂点数(%d)が不一致: ring=%s", len(uv), n, ext_id
                )

        for ring, int_elem in zip(poly.interiors, int_elems):
            int_id = int_elem.get(_GML_ID, "")
            if int_id in tex_map:
                _, uv = tex_map[int_id]
                n = len(ring.coords)
                if len(uv) >= n:
                    ring.uv = uv[:n]

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
