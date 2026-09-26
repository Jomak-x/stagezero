"""Textured Core27 actors in the existing Studio Viser scene.

The Studio's G1 actor and GLB protocol are deliberately independent.  This
adapter publishes ordinary Viser skinned meshes, using the same texture
extension already used by generated Studio characters.  A timeline's first
Core window establishes each actor's floor offset; later windows reuse the
offset and the already fitted prefix byte for byte.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from viser import _messages

from character_actor import TexturedSkinnedMeshProps
from grounded_character import GroundedCharacter
from realtime_clip import CanonicalClip


DEFAULT_ASSETS = (
    Path(__file__).resolve().parent / "assets/core-characters/civilian.glb",
    Path(__file__).resolve().parent / "assets/core-characters/ranger.glb",
)


class StudioCoreRenderer:
    """Display one or two independently skinned human actors at Core's 20 fps.

    ``set_clip`` accepts the director's complete immutable timeline.  Appending
    frames fits only the new suffix if the native prefix and actor order are
    unchanged.  Seek and repeat playback merely select cached fitted frames.
    The caller owns the G1 actor's visibility and the playback clock.
    """

    def __init__(self, server, *, asset_paths=None, floor_y: float = 0.0,
                 name_prefix: str = "/actor/core") -> None:
        if not np.isfinite(floor_y):
            raise ValueError("floor_y must be finite")
        paths = tuple(DEFAULT_ASSETS if asset_paths is None else asset_paths)
        if len(paths) != 2:
            raise ValueError("Two distinct Core character assets are required")
        if Path(paths[0]).resolve() == Path(paths[1]).resolve():
            raise ValueError("Core actors need distinct character assets")
        self.server = server
        self.floor_y = float(floor_y)
        self.characters = tuple(GroundedCharacter(path) for path in paths)
        self.handles = tuple(
            self._add_actor(f"{name_prefix}/{index}", character)
            for index, character in enumerate(self.characters)
        )
        self._clip: CanonicalClip | None = None
        self._fitted_positions: np.ndarray | None = None
        self._fitted_rotations: np.ndarray | None = None
        self._floor_offsets: tuple[float, ...] | None = None
        self._frame: int | None = None
        self._visible = False
        self._removed = False
        self.set_visible(False)

    def _add_actor(self, name: str, character: GroundedCharacter):
        identity = np.broadcast_to(np.array([1., 0., 0., 0.], np.float32),
                                   (len(character.rest), 4)).copy()
        handle = self.server.scene.add_mesh_skinned(
            name,
            np.asarray(character.vertices, np.float32),
            np.asarray(character.faces, np.uint32),
            bone_wxyzs=identity,
            bone_positions=np.asarray(character.rest, np.float32),
            skin_weights=np.asarray(character.weights, np.float32),
            color=(255, 255, 255), side="double",
            cast_shadow=True, receive_shadow=True,
        )
        try:
            original = handle._impl.props
            textured = TexturedSkinnedMeshProps(
                **{field.name: getattr(original, field.name)
                   for field in fields(_messages.SkinnedMeshProps)},
                uv=np.asarray(character.uv, np.float32),
                normals=np.asarray(character.normals, np.float32),
                texture_png=character.texture_png,
                normal_texture_png=character.normal_texture_png,
                metallic_roughness_texture_png=character.metallic_roughness_texture_png,
                **character.material,
            )
            handle._impl.props = textured
            self.server.scene._websock_interface.queue_message(
                _messages.SkinnedMeshMessage(name, textured)
            )
        except Exception:
            handle.remove()
            raise
        return handle

    @property
    def clip(self) -> CanonicalClip | None:
        return self._clip

    @property
    def frame(self) -> int | None:
        return self._frame

    @property
    def floor_offsets(self) -> tuple[float, ...] | None:
        return self._floor_offsets

    @property
    def fitted_positions(self) -> np.ndarray | None:
        return self._fitted_positions

    @property
    def fitted_rotations(self) -> np.ndarray | None:
        return self._fitted_rotations

    def _fit_actor(self, index: int, clip: CanonicalClip, start: int,
                   offset: float | None) -> tuple[np.ndarray, np.ndarray, float]:
        character = self.characters[index]
        native_positions = clip.positions[index:index + 1, start:]
        native_rotations = clip.rotations[index:index + 1, start:]
        if offset is None:
            # The first complete model window is the only calibration input.
            # A short imported clip uses all its currently available frames.
            sample = min(40, clip.frames)
            calibrated = character.clip_payload(
                clip.positions[index:index + 1, :sample],
                clip.rotations[index:index + 1, :sample],
                preserve_wrists=False, floor_y=self.floor_y,
            )
            offset = float(calibrated["character_provenance"]["floor_offsets"][0])
        payload = character.clip_payload(
            native_positions, native_rotations, preserve_wrists=False,
            preserve_feet=True, floor_y=self.floor_y, floor_offsets=[offset],
        )
        return (np.asarray(payload["fitted_positions"][0], dtype=np.float32),
                np.asarray(payload["fitted_rotations"][0], dtype=np.float32),
                offset)

    def set_clip(self, clip: CanonicalClip) -> None:
        """Fit a replacement or newly appended immutable Core timeline."""
        if self._removed:
            raise RuntimeError("Core renderer has been removed")
        if not isinstance(clip, CanonicalClip):
            raise TypeError("Expected a CanonicalClip")
        previous = self._clip
        append = (previous is not None and previous.actor_ids == clip.actor_ids
                  and clip.frames >= previous.frames
                  and np.array_equal(clip.positions[:, :previous.frames], previous.positions)
                  and np.array_equal(clip.rotations[:, :previous.frames], previous.rotations))
        if append and clip.frames == previous.frames:
            self._clip = clip
            return
        start = previous.frames if append else 0
        offsets = self._floor_offsets if append else None
        fitted = [self._fit_actor(index, clip, start,
                                  offsets[index] if offsets is not None else None)
                  for index in range(len(clip.actor_ids))]
        new_positions = np.stack([row[0] for row in fitted])
        new_rotations = np.stack([row[1] for row in fitted])
        if append:
            new_positions = np.concatenate([self._fitted_positions, new_positions], axis=1)
            new_rotations = np.concatenate([self._fitted_rotations, new_rotations], axis=1)
        new_positions.setflags(write=False)
        new_rotations.setflags(write=False)
        self._clip = clip
        self._fitted_positions = new_positions
        self._fitted_rotations = new_rotations
        self._floor_offsets = tuple(row[2] for row in fitted)
        if self._frame is not None:
            self.tick(min(self._frame, clip.frames - 1))
        else:
            self._sync_visibility()

    set_clips = set_clip

    def _sync_visibility(self) -> None:
        count = 0 if self._clip is None else len(self._clip.actor_ids)
        for index, handle in enumerate(self.handles):
            handle.visible = self._visible and index < count and self._frame is not None

    def set_visible(self, visible: bool) -> None:
        if self._removed:
            raise RuntimeError("Core renderer has been removed")
        self._visible = bool(visible)
        self._sync_visibility()

    def tick(self, frame: int) -> bool:
        """Show one fitted frame; seeking does not refit or move the floor."""
        if self._removed:
            raise RuntimeError("Core renderer has been removed")
        if self._clip is None:
            return False
        if type(frame) is not int:
            raise ValueError("frame must be an integer")
        index = max(0, min(frame, self._clip.frames - 1))
        for actor in range(len(self._clip.actor_ids)):
            positions = self._fitted_positions[actor, index]
            rotations = self._fitted_rotations[actor, index]
            quats = Rotation.from_matrix(rotations).as_quat(scalar_first=True)
            for bone, position, quat in zip(self.handles[actor].bones, positions, quats):
                bone.position = np.asarray(position, np.float32)
                bone.wxyz = np.asarray(quat, np.float32)
        self._frame = index
        self._sync_visibility()
        return True

    def update(self, clip: CanonicalClip | np.ndarray, rotations=None, *,
               frame: int | None = None, actor_ids=None) -> None:
        """Convenience for a complete clip or a single raw Core27 pose."""
        if isinstance(clip, CanonicalClip):
            self.set_clip(clip)
            self.tick(0 if frame is None else frame)
            return
        positions = np.asarray(clip)
        matrices = np.asarray(rotations)
        if positions.ndim == 2:
            positions, matrices = positions[None], matrices[None]
        if positions.ndim != 3 or positions.shape[1:] != (27, 3):
            raise ValueError("Expected one or two Core27 poses")
        ids = tuple(actor_ids) if actor_ids is not None else tuple(f"actor_{i}" for i in range(len(positions)))
        self.set_clip(CanonicalClip.from_arrays(positions[:, None], matrices[:, None],
                                                actor_ids=ids, source="ardy_core"))
        self.tick(0)

    def actor_root(self, actor_id: str | int = 0) -> tuple[float, float, float] | None:
        if self._clip is None or self._frame is None:
            return None
        index = actor_id if type(actor_id) is int else self._clip.actor_ids.index(actor_id)
        if not 0 <= index < len(self._clip.actor_ids):
            raise IndexError("Unknown Core actor")
        return tuple(float(value) for value in self._fitted_positions[index, self._frame, 0])

    def remove(self) -> None:
        if self._removed:
            return
        for handle in self.handles:
            handle.remove()
        self._removed = True
        self._clip = None
        self._fitted_positions = None
        self._fitted_rotations = None
