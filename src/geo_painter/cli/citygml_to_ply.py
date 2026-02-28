"""CityGML → PLY 変換 CLI

使い方:
    # お台場プリセットで変換
    uv run citygml-to-ply convert=odaiba

    # 原点を明示指定して変換
    uv run citygml-to-ply convert=odaiba convert.origin.lat=35.63 convert.origin.lon=139.77

    # 入出力パスを上書き
    uv run citygml-to-ply convert=odaiba \\
        convert.input_dir=/path/to/data \\
        convert.output_path=/tmp/output.ply

    # 地物種別を限定
    uv run citygml-to-ply convert=odaiba 'convert.feature_types=[building]'
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from geo_painter.converter import CityGmlToPlyConverter

# editableインストール前提で conf/ をパッケージルートから解決
_CONF_DIR = str(Path(__file__).parents[3] / "conf")


@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="convert_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("設定:\n%s", OmegaConf.to_yaml(cfg))

    converter = CityGmlToPlyConverter(cfg)
    output_path = converter.run()

    logger.info("完了: %s", output_path)


if __name__ == "__main__":
    main()
