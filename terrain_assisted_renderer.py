"""Render explicitly solved character-rig poses without another retarget/IK pass."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from studio_core_renderer import DEFAULT_ASSETS, StudioCoreRenderer


class TerrainAssistedRenderer(StudioCoreRenderer):
    """The input is a separate mesh17 presentation, never native Core history."""

    def __init__(self, server, *, asset_paths=None, name_prefix="/terrain-assisted"):
        paths = tuple(DEFAULT_ASSETS if asset_paths is None else asset_paths)
        self.rig_asset_sha256 = tuple(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths)
        super().__init__(server, asset_paths=paths, name_prefix=name_prefix)

    def set_clip(self, clip):
        raise TypeError("Use set_presentation for explicitly solved mesh-rig arrays")

    def set_presentation(self, presentation, *, actor_ids):
        if self._removed:
            raise RuntimeError("Terrain renderer has been removed")
        ids = tuple(actor_ids)
        if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
            raise ValueError("Expected one or two distinct actor IDs")
        positions = np.asarray(presentation.positions, dtype=np.float32)
        rotations = np.asarray(presentation.rotations, dtype=np.float32)
        if (positions.ndim != 4 or positions.shape[0] != len(ids)
                or positions.shape[2:] != (17, 3) or not 1 <= positions.shape[1] <= 15000
                or rotations.shape != positions.shape[:3]+(3, 3)
                or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
            raise ValueError("Expected finite mesh17 presentation arrays")
        if (not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=.03)
                or not np.allclose(np.linalg.det(rotations), 1, atol=.03)):
            raise ValueError("Presentation rotations must be proper orthonormal matrices")
        if tuple(presentation.rig_asset_sha256) != self.rig_asset_sha256[:len(ids)]:
            raise ValueError("Presentation was solved for different character assets")
        self._fitted_positions = np.array(positions, copy=True)
        self._fitted_rotations = np.array(rotations, copy=True)
        self._clip = SimpleNamespace(frames=positions.shape[1], actor_ids=ids)
        self._floor_offsets = (0.,)*len(ids)
        self._fitting_provenance = {
            "source": "explicit terrain-assisted mesh17 presentation",
            "rig_asset_sha256": list(presentation.rig_asset_sha256),
            "retarget_applied": False, "additional_foot_ik": False,
            "root_translation_applied": False, "native_history": False,
        }
        self.tick(min(self._frame or 0, positions.shape[1]-1))
