"""座標変換モジュール

EPSG:6668（JGD2011 地理座標系: lat/lon/高さ）を
ローカルENU（East-North-Up、単位: m）に変換する。

等距離方位図法（aeqd）を用いて参照点を原点とする平面座標に投影する。
"""

from __future__ import annotations

import logging

import numpy as np
from pyproj import Transformer

from geo_painter.citygml.models import Ring

logger = logging.getLogger(__name__)

# EPSG:6668 は JGD2011 地理座標系（緯度・経度・楕円体高）
# +proj=aeqd は等距離方位図法（局所ENU近似として利用）
_WGS84_EPSG = "EPSG:4326"


class CoordinateTransformer:
    """地理座標からローカルENU座標への変換クラス

    参照点 (origin_lat, origin_lon) を原点とする平面座標（m単位）に投影する。
    origin を省略した場合は最初の transform_ring() 呼び出し時に遅延初期化する。

    Args:
        origin_lat: 参照点の緯度（度）
        origin_lon: 参照点の経度（度）
    """

    def __init__(
        self,
        origin_lat: float | None = None,
        origin_lon: float | None = None,
    ) -> None:
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._transformer: Transformer | None = None

        if origin_lat is not None and origin_lon is not None:
            self._init_transformer(origin_lat, origin_lon)

    def _init_transformer(self, lat: float, lon: float) -> None:
        """等距離方位図法のトランスフォーマーを初期化する"""
        proj_str = (
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        self._transformer = Transformer.from_crs(
            _WGS84_EPSG,
            proj_str,
            always_xy=True,  # 入力を (lon, lat) 順に固定
        )
        self._origin_lat = lat
        self._origin_lon = lon
        logger.debug("座標変換原点を設定: lat=%.6f, lon=%.6f", lat, lon)

    def transform_ring(self, ring: Ring) -> np.ndarray:
        """Ring の座標をローカルENU座標（m）に変換する

        Args:
            ring: Ring オブジェクト (lat, lon, height) の配列

        Returns:
            変換後の座標配列 shape=(N, 3)、単位 m
        """
        coords = ring.coords  # (N, 3): lat, lon, height

        # 遅延初期化: 最初の頂点を参照点とする
        if self._transformer is None:
            first = coords[0]
            logger.info("原点を最初の頂点から自動設定: lat=%.6f, lon=%.6f", first[0], first[1])
            self._init_transformer(float(first[0]), float(first[1]))

        assert self._transformer is not None

        lats = coords[:, 0]
        lons = coords[:, 1]
        heights = coords[:, 2]

        # pyproj は always_xy=True のとき (lon, lat) 順で入力を期待する
        xs, ys = self._transformer.transform(lons, lats)

        result = np.column_stack([xs, ys, heights]).astype(np.float64)
        return result

    @property
    def origin_lat(self) -> float | None:
        return self._origin_lat

    @property
    def origin_lon(self) -> float | None:
        return self._origin_lon
