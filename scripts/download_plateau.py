#!/usr/bin/env python3
"""PLATEAUデータダウンロードスクリプト

使い方:
    # デフォルト設定（お台場）でダウンロード
    python scripts/download_plateau.py

    # リソース一覧だけ確認（ダウンロードしない）
    python scripts/download_plateau.py +info=true

    # 別のプリセットに切り替え（conf/plateau/ に追加したもの）
    python scripts/download_plateau.py plateau=odaiba

    # 設定値を上書き
    python scripts/download_plateau.py output.base_dir=/tmp/plateau

    # データセットIDを直接指定
    python scripts/download_plateau.py \\
        'plateau.dataset_ids=[plateau-13113-shibuya-ku-2023]' \\
        'plateau.file_types=[CityGML]'

    # 全ファイル種別をダウンロード（CityGML以外も含む）
    python scripts/download_plateau.py 'plateau.file_types=[]'
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# confディレクトリをパッケージルート基準で解決するため sys.path を調整
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geo_painter.plateau import PlateauDownloader

# Hydraのconfig_pathはこのスクリプトからの相対パス
CONFIG_DIR = str(Path(__file__).parent.parent / "conf")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("設定:\n%s", OmegaConf.to_yaml(cfg))

    downloader = PlateauDownloader(cfg)

    # +info=true の場合はリソース一覧だけ表示して終了
    if cfg.get("info", False):
        logger.info("=== リソース一覧 (ダウンロードなし) ===")
        downloader.list_resources()
        return

    results = downloader.run()

    logger.info("=== ダウンロード結果 ===")
    total = 0
    for dataset_id, paths in results.items():
        logger.info("[%s] %d ファイル", dataset_id, len(paths))
        for p in paths:
            logger.info("  %s", p)
        total += len(paths)
    logger.info("合計: %d ファイル", total)


if __name__ == "__main__":
    main()
