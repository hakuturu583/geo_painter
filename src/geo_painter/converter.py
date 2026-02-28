"""CityGML → PLY 変換コンバーター

Scanner → Parser → Transformer → Triangulate → PLY書き出しの
パイプラインを実装するファサードクラス。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from geo_painter.citygml.models import FeatureType, Geometry
from geo_painter.citygml.parser import CityGMLParser, CityGMLScanner
from geo_painter.mesh.transform import CoordinateTransformer
from geo_painter.mesh.triangulate import triangulate_geometry
from geo_painter.plateau import PlateauDownloader
from geo_painter.plateau.downloader import _extract_version, _VERSION_RE

logger = logging.getLogger(__name__)


class CityGmlToPlyConverter:
    """CityGML（ZIP）を読み込みPLYファイルを出力するコンバーター

    Args:
        cfg: Hydra 設定オブジェクト
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def run(self) -> list[Path]:
        """変換パイプラインを実行してPLYファイルを書き出す

        source=plateau の場合はデータセットごとに1ファイル出力する。
        出力先ディレクトリは output_path の親ディレクトリ。

        Returns:
            出力PLYファイルのパスリスト
        """
        cfg = self._cfg
        input_dir = Path(cfg.convert.input_dir)
        source = cfg.convert.get("source", "file")
        output_path = Path(cfg.convert.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 処理対象の地物種別
        target_types: set[FeatureType] = set()
        for ft_name in cfg.convert.feature_types:
            try:
                target_types.add(FeatureType(ft_name))
            except ValueError:
                logger.warning("不明な地物種別: %s", ft_name)

        # 座標変換器を初期化
        origin = cfg.convert.get("origin", None)
        if origin is not None:
            transformer = CoordinateTransformer(
                origin_lat=float(origin.lat),
                origin_lon=float(origin.lon),
            )
            logger.info("原点: lat=%.6f, lon=%.6f", origin.lat, origin.lon)
        else:
            transformer = CoordinateTransformer()
            logger.info("原点は最初の頂点から自動設定されます")

        if source == "plateau":
            return self._run_plateau(input_dir, output_path, cfg, target_types, transformer)

        # source == "file": 単一 PLY に統合
        zip_files = self._resolve_file_zips(input_dir)
        if not zip_files:
            raise FileNotFoundError(f"ZIPファイルが見つかりません: {input_dir}")
        logger.info("%d 個の ZIP ファイルを処理します", len(zip_files))

        vertices, faces, colors = self._collect_geometry(zip_files, target_types, transformer)
        self._write_ply(output_path, vertices, faces, colors, transformer)
        logger.info("PLY 出力完了: %s", output_path)
        return [output_path]

    def _run_plateau(
        self,
        input_dir: Path,
        output_path: Path,
        cfg: DictConfig,
        target_types: set[FeatureType],
        transformer: CoordinateTransformer,
    ) -> list[Path]:
        """source=plateau のときにデータセットごとに PLY を書き出す

        出力先は output_path の親ディレクトリ。
        ファイル名は {dataset_id}.ply。

        Args:
            input_dir: ダウンロードキャッシュのベースディレクトリ
            output_path: 出力先（親ディレクトリを出力ディレクトリとして使用）
            cfg: Hydra 設定
            target_types: 処理対象の地物種別
            transformer: 座標変換器

        Returns:
            出力された PLY ファイルパスのリスト
        """
        plateau_cfg = cfg.convert.plateau
        dataset_ids: list[str] = list(plateau_cfg.dataset_ids)
        output_dir = output_path.parent

        # 未キャッシュのデータセットをダウンロード
        missing: list[str] = []
        for dataset_id in dataset_ids:
            dataset_dir = input_dir / dataset_id
            cached = list(dataset_dir.glob("*.zip")) if dataset_dir.exists() else []
            if cached:
                logger.info("キャッシュ確認 OK [%s]: %d 個の ZIP", dataset_id, len(cached))
            else:
                logger.info("キャッシュなし [%s]: ダウンロードが必要", dataset_id)
                missing.append(dataset_id)

        if missing:
            logger.info("%d 個のデータセットをダウンロードします", len(missing))
            self._download_datasets(missing, input_dir, cfg)

        # データセットごとに PLY を生成
        results: list[Path] = []
        for dataset_id in tqdm(dataset_ids, desc="データセット処理", unit="dataset"):
            zip_files = self._pick_latest_zip_versions(
                list((input_dir / dataset_id).glob("*.zip"))
            )
            if not zip_files:
                logger.warning("ZIPが見つかりません: %s", dataset_id)
                continue

            logger.info("[%s] %d 個の ZIP ファイルを処理します", dataset_id, len(zip_files))
            vertices, faces, colors = self._collect_geometry(zip_files, target_types, transformer)

            ply_path = output_dir / f"{dataset_id}.ply"
            self._write_ply(ply_path, vertices, faces, colors, transformer)
            logger.info("[%s] PLY 出力完了: %s", dataset_id, ply_path)
            results.append(ply_path)

        return results

    def _collect_geometry(
        self,
        zip_files: list[Path],
        target_types: set[FeatureType],
        transformer: CoordinateTransformer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ZIP ファイル群からジオメトリを収集して結合する

        Args:
            zip_files: 処理対象の ZIP ファイルリスト
            target_types: 処理対象の地物種別
            transformer: 座標変換器

        Returns:
            (vertices, faces, colors) のタプル。ジオメトリがない場合は空配列。
        """
        all_vertices: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        vertex_offset = 0

        parser = CityGMLParser()

        for zip_path in tqdm(zip_files, desc="ZIP処理", unit="zip"):
            logger.info("ZIP 処理中: %s", zip_path.name)
            scanner = CityGMLScanner(zip_path)

            gml_files = [
                (ft, s)
                for ft, s in scanner.iter_files()
                if ft in target_types
            ]
            for feature_type, stream in tqdm(
                gml_files,
                desc=zip_path.name,
                unit="gml",
                leave=False,
            ):
                geometries = parser.parse(stream, feature_type)

                for geom in tqdm(
                    geometries,
                    desc=f"  {feature_type.value}",
                    unit="geom",
                    leave=False,
                ):
                    verts, faces = triangulate_geometry(geom, transformer)
                    if len(faces) == 0:
                        continue

                    color = np.array(feature_type.color, dtype=np.uint8)
                    colors = np.tile(color, (len(verts), 1))

                    all_vertices.append(verts)
                    all_faces.append(faces + vertex_offset)
                    all_colors.append(colors)
                    vertex_offset += len(verts)

        if not all_vertices:
            logger.warning("変換できるジオメトリがありませんでした")
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.uint8),
            )

        return (
            np.concatenate(all_vertices, axis=0),
            np.concatenate(all_faces, axis=0),
            np.concatenate(all_colors, axis=0),
        )

    def _resolve_file_zips(self, input_dir: Path) -> list[Path]:
        """source=file のとき input_dir 直下の *.zip を返す"""
        zip_files = self._pick_latest_zip_versions(sorted(input_dir.glob("*.zip")))
        logger.info("source=file: %d 個の ZIP を検出 (%s)", len(zip_files), input_dir)
        return zip_files

    def _download_datasets(
        self, dataset_ids: list[str], input_dir: Path, cfg: DictConfig
    ) -> None:
        """PlateauDownloader を使って指定データセットをダウンロードする

        Args:
            dataset_ids: ダウンロードするデータセット ID のリスト
            input_dir: ダウンロード先ベースディレクトリ
            cfg: Hydra 設定（convert.plateau.* を参照）
        """
        plateau_cfg = cfg.convert.plateau
        download_cfg = OmegaConf.select(plateau_cfg, "download", default=None)
        chunk_size = download_cfg.chunk_size if download_cfg else 8192
        timeout = download_cfg.timeout if download_cfg else 60

        downloader_cfg = OmegaConf.create(
            {
                "plateau": {
                    "dataset_ids": dataset_ids,
                    "file_types": list(plateau_cfg.file_types),
                },
                "output": {
                    "base_dir": str(input_dir),
                },
                "download": {
                    "chunk_size": chunk_size,
                    "timeout": timeout,
                },
            }
        )
        downloader = PlateauDownloader(downloader_cfg)
        results = downloader.run()
        total = sum(len(paths) for paths in results.values())
        logger.info("ダウンロード完了: 合計 %d ファイル", total)

    @staticmethod
    def _pick_latest_zip_versions(zip_files: list[Path]) -> list[Path]:
        """ZIPファイル名の ``（vN）`` 表記で最新バージョンのみに絞り込む。

        同一グループ（バージョン表記を除いたファイル名が同じ）内で最大バージョン
        番号のファイルだけを残す。バージョン表記のないファイルはそのまま保持する。
        """
        groups: dict[tuple[Path, str], list[Path]] = {}
        for p in zip_files:
            # 親ディレクトリ単位でグループ化（異なるデータセットの同名ZIPを混在させない）
            key = (p.parent, _VERSION_RE.sub("", p.stem).strip())
            groups.setdefault(key, []).append(p)

        result: list[Path] = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
            else:
                latest = max(group, key=lambda p: _extract_version(p.stem))
                for p in group:
                    if p is not latest:
                        logger.info(
                            "旧バージョンZIPをスキップ: %s（最新: %s）",
                            p.name,
                            latest.name,
                        )
                result.append(latest)
        return sorted(result)

    @staticmethod
    def _write_ply(
        output_path: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        transformer: CoordinateTransformer,
    ) -> None:
        """バイナリ little-endian PLY ファイルを書き出す

        Args:
            output_path: 出力ファイルパス
            vertices: 頂点座標 shape=(V, 3)
            faces: 面インデックス shape=(F, 3)
            colors: 頂点色 shape=(V, 3) uint8
            transformer: 座標変換器（参照点情報をコメントに記録）
        """
        # 頂点要素
        vertex_data = np.zeros(
            len(vertices),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        if len(vertices) > 0:
            vertex_data["x"] = vertices[:, 0]
            vertex_data["y"] = vertices[:, 1]
            vertex_data["z"] = vertices[:, 2]
            vertex_data["red"] = colors[:, 0]
            vertex_data["green"] = colors[:, 1]
            vertex_data["blue"] = colors[:, 2]

        vertex_elem = PlyElement.describe(vertex_data, "vertex")

        # 面要素
        if len(faces) > 0:
            face_data = np.empty(len(faces), dtype=[("vertex_indices", "O")])
            for i, face in enumerate(faces):
                face_data["vertex_indices"][i] = face.astype(np.int32)
        else:
            face_data = np.empty(0, dtype=[("vertex_indices", "O")])

        face_elem = PlyElement.describe(face_data, "face")

        # コメントに参照点情報を記録
        comments: list[str] = ["Generated by geo-painter citygml-to-ply"]
        if transformer.origin_lat is not None and transformer.origin_lon is not None:
            comments.append(
                f"origin lat={transformer.origin_lat:.6f} lon={transformer.origin_lon:.6f}"
            )

        ply = PlyData(
            [vertex_elem, face_elem],
            text=False,
            byte_order="<",  # little-endian
            comments=comments,
        )
        ply.write(str(output_path))
