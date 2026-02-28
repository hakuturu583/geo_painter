"""PLATEAUデータダウンロードCLI

使い方:
    # デフォルト設定（お台場）でダウンロード
    uv run plateau-download

    # リソース一覧だけ確認（ダウンロードしない）
    uv run plateau-download +info=true

    # 別のプリセットに切り替え（conf/plateau/ に追加したもの）
    uv run plateau-download plateau=odaiba

    # 設定値を上書き
    uv run plateau-download output.base_dir=/tmp/plateau

    # データセットIDを直接指定
    uv run plateau-download \\
        'plateau.dataset_ids=[plateau-13113-shibuya-ku-2023]' \\
        'plateau.file_types=[CityGML]'

    # 全ファイル種別をダウンロード（CityGML以外も含む）
    uv run plateau-download 'plateau.file_types=[]'
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from geo_painter.plateau import PlateauDownloader

# editableインストール前提で conf/ をパッケージルートから解決
_CONF_DIR = str(Path(__file__).parents[3] / "conf")


@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
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
