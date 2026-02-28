"""CityGML データモデル定義"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np


class FeatureType(str, Enum):
    """地物種別と対応する表示色 (R, G, B)"""

    BUILDING = "building"
    ROAD = "road"
    TERRAIN = "terrain"

    @property
    def color(self) -> Tuple[int, int, int]:
        """地物種別に対応するRGB色を返す"""
        _COLORS = {
            FeatureType.BUILDING: (255, 200, 100),
            FeatureType.ROAD: (180, 180, 180),
            FeatureType.TERRAIN: (120, 180, 80),
        }
        return _COLORS[self]

    @property
    def udx_dir(self) -> str:
        """ZIPアーカイブ内のudxサブディレクトリ名を返す"""
        _DIRS = {
            FeatureType.BUILDING: "bldg",
            FeatureType.ROAD: "tran",
            FeatureType.TERRAIN: "dem",
        }
        return _DIRS[self]


@dataclass
class Ring:
    """GMLのLinearRing（外輪・内輪共通）を表すデータクラス

    Attributes:
        coords: 頂点座標配列 shape=(N, 3)、各列は (lat, lon, height)
                末尾の重複点（閉合点）は除去済み
    """

    coords: np.ndarray  # shape (N, 3), dtype float64


@dataclass
class GmlPolygon:
    """GMLのPolygonを表すデータクラス

    Attributes:
        exterior: 外輪リング
        interiors: 内輪リングのリスト（穴）
    """

    exterior: Ring
    interiors: list[Ring] = field(default_factory=list)


@dataclass
class Geometry:
    """1つの地物フィーチャーのジオメトリを表すデータクラス

    Attributes:
        feature_type: 地物種別
        feature_id: GML ID
        lod: LODレベル（整数）
        polygons: ポリゴンリスト
    """

    feature_type: FeatureType
    feature_id: str
    lod: int
    polygons: list[GmlPolygon] = field(default_factory=list)
