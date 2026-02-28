"""ポリゴン三角分割モジュール

earcut アルゴリズムを使ってポリゴンを三角形メッシュに分割する。
earcut が失敗した場合や未インストールの場合はファン三角分割にフォールバックする。
"""

from __future__ import annotations

import logging

import numpy as np

from geo_painter.citygml.models import Geometry
from geo_painter.mesh.transform import CoordinateTransformer

logger = logging.getLogger(__name__)

# mapbox_earcut の可否をインポート時に確認する
try:
    import mapbox_earcut as earcut

    _EARCUT_AVAILABLE = True
except ImportError:
    _EARCUT_AVAILABLE = False
    logger.warning(
        "mapbox-earcut が未インストールです。ファン三角分割にフォールバックします。"
    )


def triangulate_geometry(
    geom: Geometry,
    transformer: CoordinateTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometry をローカルENU座標で三角分割する

    Args:
        geom: 三角分割対象の Geometry
        transformer: 座標変換オブジェクト

    Returns:
        (vertices, faces) のタプル
        vertices: float32 配列 shape=(V, 3)
        faces:    int32  配列 shape=(F, 3)
    """
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for polygon in geom.polygons:
        # exterior を ENU 変換
        ext_enu = transformer.transform_ring(polygon.exterior)

        # interior を ENU 変換
        int_enus: list[np.ndarray] = []
        for interior in polygon.interiors:
            int_enus.append(transformer.transform_ring(interior))

        # 三角分割
        verts, faces = _triangulate_polygon(ext_enu, int_enus)
        if len(faces) == 0:
            continue

        all_vertices.append(verts)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(verts)

    if not all_vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)
    return vertices, faces


def _triangulate_polygon(
    exterior: np.ndarray,
    interiors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """1つのポリゴンを三角分割する

    earcut → fan triangulation のフォールバック順で試みる。

    Args:
        exterior: 外輪座標 shape=(N, 3)
        interiors: 内輪座標リスト

    Returns:
        (vertices, faces) のタプル
    """
    if _EARCUT_AVAILABLE:
        result = _try_earcut(exterior, interiors)
        if result is not None:
            return result
        logger.debug("earcut 失敗、ファン三角分割にフォールバック")

    return _fan_triangulate(exterior)


def _try_earcut(
    exterior: np.ndarray,
    interiors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """mapbox_earcut を使ってポリゴンを三角分割する

    XY平面でのみ earcut を実行し、Z座標は外輪から引き継ぐ。

    Args:
        exterior: 外輪座標 shape=(N, 3)
        interiors: 内輪座標リスト

    Returns:
        成功時: (vertices, faces)、失敗時: None
    """
    try:
        # 全頂点を結合 (exterior + interiors)
        all_rings = [exterior] + interiors
        vertices = np.concatenate(all_rings, axis=0)

        # hole_indices: 各 interior リングの開始インデックス
        hole_indices: list[int] = []
        offset = len(exterior)
        for interior in interiors:
            hole_indices.append(offset)
            offset += len(interior)

        # earcut は 2D の平面座標で実行（XY 平面を使用）
        xy = vertices[:, :2].flatten().astype(np.float64)
        rings = np.array(hole_indices, dtype=np.uint32) if hole_indices else np.array([], dtype=np.uint32)

        indices = earcut.triangulate_float64(xy, rings, 2)

        if len(indices) == 0 or len(indices) % 3 != 0:
            return None

        faces = indices.reshape(-1, 3).astype(np.int32)
        return vertices.astype(np.float32), faces

    except Exception as exc:
        logger.debug("earcut 例外: %s", exc)
        return None


def _fan_triangulate(exterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ファン三角分割（凸多角形・単純多角形向けフォールバック）

    Args:
        exterior: 外輪座標 shape=(N, 3)

    Returns:
        (vertices, faces)
    """
    n = len(exterior)
    if n < 3:
        return exterior.astype(np.float32), np.zeros((0, 3), dtype=np.int32)

    # 0番頂点を中心とするファン分割
    faces = np.array(
        [[0, i, i + 1] for i in range(1, n - 1)],
        dtype=np.int32,
    )
    return exterior.astype(np.float32), faces
