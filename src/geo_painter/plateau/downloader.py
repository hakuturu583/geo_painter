"""PLATEAUデータダウンローダー

G空間情報センター（https://www.geospatial.jp/）のCKAN APIを使って
PLATEAU 3D都市モデルをダウンロードするモジュール。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import requests
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

CKAN_BASE_URL = "https://www.geospatial.jp/ckan/api/3/action"

# バージョン表記のパターン: （v4）/ (v4) など
_VERSION_RE = re.compile(r"\s*[（(]v(\d+)[）)]\s*")


def _extract_version(name: str) -> int:
    """名前文字列から `（vN）` 形式のバージョン番号を抽出する。未検出は 0。"""
    m = _VERSION_RE.search(name)
    return int(m.group(1)) if m else 0


def pick_latest_versions(
    resources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """同種別リソースに複数バージョンがある場合、最新バージョンのみ返す。

    ``（vN）`` / ``(vN)`` を除去した名前でグループ化し、同一グループ内で
    最大バージョン番号のリソースのみを残す。バージョン表記のないリソースは
    単独グループとしてそのまま保持する。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in resources:
        key = _VERSION_RE.sub("", r.get("name", "")).strip()
        groups.setdefault(key, []).append(r)

    result: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            latest = max(group, key=lambda r: _extract_version(r.get("name", "")))
            for r in group:
                if r is not latest:
                    logger.info(
                        "旧バージョンをスキップ: %s（最新: %s）",
                        r.get("name", "?"),
                        latest.get("name", "?"),
                    )
            result.append(latest)
    return result


class PlateauDownloader:
    """PLATEAU 3D都市モデルのダウンローダー。

    G空間情報センターのCKAN APIを使い、指定した自治体の
    PLATEAUデータを検索・ダウンロードする。
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "geo-painter/0.1.0"})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_resources(self, dataset_id: str) -> list[dict[str, Any]]:
        """データセットIDからリソース一覧を取得する。

        Args:
            dataset_id: G空間情報センター上のデータセットID
                例: "plateau-13103-minato-ku-2023"

        Returns:
            リソース情報のリスト（name, url, format 等を含む dict）
        """
        url = f"{CKAN_BASE_URL}/package_show"
        resp = self.session.get(
            url,
            params={"id": dataset_id},
            timeout=self.cfg.download.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        if not result.get("success"):
            raise RuntimeError(f"CKAN API error for dataset '{dataset_id}': {result}")
        return result["result"]["resources"]

    def filter_resources(
        self,
        resources: list[dict[str, Any]],
        file_types: list[str],
    ) -> list[dict[str, Any]]:
        """ファイル種別で絞り込む。

        Args:
            resources: `fetch_resources` の返り値
            file_types: 欲しいファイル種別のリスト
                例: ["CityGML", "3D Tiles"]
                空リストの場合はすべてを返す。

        Returns:
            絞り込まれたリソースのリスト
        """
        if not file_types:
            return pick_latest_versions(resources)

        lower_types = {ft.lower() for ft in file_types}
        filtered = [
            r
            for r in resources
            if any(ft in r.get("name", "").lower() for ft in lower_types)
            or r.get("format", "").lower() in lower_types
        ]
        return pick_latest_versions(filtered)

    def download_resource(
        self,
        resource: dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """リソースを1ファイルダウンロードする。

        Args:
            resource: `fetch_resources` が返す個々のリソース dict
            output_dir: 保存先ディレクトリ

        Returns:
            保存したファイルのパス
        """
        url: str = resource["url"]
        name: str = resource.get("name", "") or Path(url).name
        # 拡張子を補完
        suffix = Path(url.split("?")[0]).suffix
        if suffix and not name.endswith(suffix):
            name = name + suffix

        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / name

        if dest.exists():
            logger.info("スキップ（既存）: %s", dest)
            return dest

        logger.info("ダウンロード開始: %s -> %s", url, dest)
        with self.session.get(
            url,
            stream=True,
            timeout=self.cfg.download.timeout,
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            with (
                dest.open("wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=name,
                    leave=False,
                ) as bar,
            ):
                for chunk in resp.iter_content(
                    chunk_size=self.cfg.download.chunk_size
                ):
                    fh.write(chunk)
                    bar.update(len(chunk))

        logger.info("ダウンロード完了: %s", dest)
        return dest

    def download_dataset(
        self,
        dataset_id: str,
        output_dir: Path,
        file_types: list[str] | None = None,
    ) -> list[Path]:
        """データセット全体をダウンロードする。

        Args:
            dataset_id: G空間情報センター上のデータセットID
            output_dir: 保存先ディレクトリ
            file_types: 欲しいファイル種別（None または空リストで全件）

        Returns:
            ダウンロードしたファイルパスのリスト
        """
        file_types = file_types or []
        logger.info("データセット取得: %s", dataset_id)
        resources = self.fetch_resources(dataset_id)
        logger.info("リソース数（全件）: %d", len(resources))

        filtered = self.filter_resources(resources, file_types)
        logger.info(
            "リソース数（フィルタ後 %s）: %d",
            file_types if file_types else "全件",
            len(filtered),
        )

        saved: list[Path] = []
        dataset_dir = output_dir / dataset_id
        for resource in filtered:
            try:
                path = self.download_resource(resource, dataset_dir)
                saved.append(path)
            except Exception:
                logger.exception(
                    "ダウンロード失敗: %s", resource.get("url", "unknown")
                )

        return saved

    def list_resources(self) -> dict[str, list[dict[str, Any]]]:
        """設定内の全データセットのリソース一覧を返す（ダウンロードなし）。

        Returns:
            {dataset_id: [resource_dict, ...]} の辞書
        """
        plateau_cfg = self.cfg.plateau
        file_types: list[str] = list(plateau_cfg.get("file_types", []))

        all_resources: dict[str, list[dict[str, Any]]] = {}
        for dataset_id in plateau_cfg.dataset_ids:
            resources = self.fetch_resources(dataset_id)
            filtered = self.filter_resources(resources, file_types)
            all_resources[dataset_id] = filtered
            logger.info(
                "[%s] リソース数: %d 件 (フィルタ前: %d 件)",
                dataset_id,
                len(filtered),
                len(resources),
            )
            for r in filtered:
                logger.info("  - %s  url=%s", r.get("name", "?"), r.get("url", "?"))
        return all_resources

    def run(self) -> dict[str, list[Path]]:
        """設定に従いすべてのデータセットをダウンロードする。

        Returns:
            {dataset_id: [保存パス, ...]} の辞書
        """
        plateau_cfg = self.cfg.plateau
        output_base = Path(self.cfg.output.base_dir)
        file_types: list[str] = list(plateau_cfg.get("file_types", []))

        results: dict[str, list[Path]] = {}
        for dataset_id in plateau_cfg.dataset_ids:
            paths = self.download_dataset(
                dataset_id=dataset_id,
                output_dir=output_base,
                file_types=file_types,
            )
            results[dataset_id] = paths
            logger.info(
                "完了 [%s]: %d ファイル", dataset_id, len(paths)
            )

        return results
