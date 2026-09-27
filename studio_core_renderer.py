"""Textured Core27 actors in the existing Studio Viser scene.

The Studio's G1 actor and GLB protocol are deliberately independent.  This
adapter publishes ordinary Viser skinned meshes, using the same texture
extension already used by generated Studio characters.  A timeline's first
Core window establishes each actor's floor offset; later windows reuse the
offset and the already fitted prefix byte for byte.
"""

from __future__ import annotations

from copy import deepcopy
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
                 name_prefix: str = "/actor/core", paired_retarget: bool = False) -> None:
        if type(paired_retarget) is not bool:
            raise ValueError("paired_retarget must be a boolean")
        self.paired_retarget = paired_retarget
        self._fitting_provenance: dict | None = None
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
        self._paired_root_height_offset = None
        if self.paired_retarget:
            offsets = [character.root_height_offset for character in self.characters]
            # The body and its hand targets must receive the same canonical
            # height correction. Unshifted world wrists made resting arms lift
            # relative to the lowered mesh torso. A shared correction preserves
            # pair contact; differently proportioned root mappings need their
            # own reviewed retargeting strategy instead of silent drift.
            if not np.allclose(offsets, offsets[0], atol=1e-7, rtol=0):
                raise ValueError("Paired retargeting requires matching character root-height offsets")
            self._paired_root_height_offset = float(offsets[0])
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

    @property
    def fitting_provenance(self) -> dict | None:
        """Disclosed fitting measurements, never a claim of visible contact."""
        return deepcopy(self._fitting_provenance)

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
            native_positions, native_rotations, preserve_wrists=self.paired_retarget,
            wrist_target_space="retargeted_root",
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
        if self.paired_retarget and len(clip.actor_ids) != 2:
            raise ValueError("Paired retargeting requires exactly two actors")
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
        if self.paired_retarget and offsets is None:
            # One pair translation retains the model's relative wrist heights.
            # Use the higher of the two support estimates to avoid lowering
            # either actor below its calibrated floor. No actor is moved apart.
            estimates = []
            for index, character in enumerate(self.characters):
                sample = min(40, clip.frames)
                payload = character.clip_payload(
                    clip.positions[index:index + 1, :sample],
                    clip.rotations[index:index + 1, :sample],
                    preserve_wrists=True, wrist_target_space="retargeted_root",
                    floor_y=self.floor_y,
                )
                estimates.append(float(payload["character_provenance"]["floor_offsets"][0]))
            offsets = (max(estimates),) * 2
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
        self._fitting_provenance = self._measure_fitting(clip)
        if self._frame is not None:
            self.tick(min(self._frame, clip.frames - 1))
        else:
            self._sync_visibility()

    def _measure_fitting(self, clip: CanonicalClip) -> dict:
        result = {
            "paired_retarget": self.paired_retarget,
            "preserve_native_wrists": self.paired_retarget,
            "floor_offsets_m": list(self._floor_offsets),
            "floor_method": ("common maximum of initial actor support offsets; reused on append"
                             if self.paired_retarget else "independent initial actor support offsets; reused on append"),
        }
        if not self.paired_retarget:
            return result
        wrists = [self.characters[0].core_idx[name] for name in ("LeftHand", "RightHand")]
        targets = np.array(clip.positions[:, :, wrists], dtype=float, copy=True)
        targets[:, :, :, 1] += self._floor_offsets[0] + self._paired_root_height_offset
        fitted = self._fitted_positions[:, :, [5, 8]]
        errors = np.linalg.norm(fitted - targets, axis=-1)
        result.update({
            "wrist_target_space": "native world plus shared canonical root-height correction and common floor translation",
            "shared_root_height_correction_m": self._paired_root_height_offset,
            "wrist_target_error_m": {
                "mean": float(errors.mean()), "max": float(errors.max()),
                "fraction_over_5mm": float(np.mean(errors > .005)),
                "per_actor_max": [float(row.max()) for row in errors],
            },
            "limitations": "Endpoint IK follows frame-native wrists with the same shared height correction as the body; no shared hand lock, finger pose, mesh contact, or collision guarantee. Common floor shift may leave one sole elevated.",
        })
        pairs = {}
        for first, name_a in enumerate(("left", "right")):
            for second, name_b in enumerate(("left", "right")):
                native_gap = np.linalg.norm(targets[0, :, first] - targets[1, :, second], axis=-1)
                rendered_gap = np.linalg.norm(fitted[0, :, first] - fitted[1, :, second], axis=-1)
                pairs[f"{name_a}_{name_b}"] = {
                    "native_min_m": float(native_gap.min()),
                    "rendered_min_m": float(rendered_gap.min()),
                    "native_frames_under_15cm": int(np.sum(native_gap < .15)),
                    "rendered_frames_under_15cm": int(np.sum(rendered_gap < .15)),
                    "gap_drift_max_m": float(np.max(np.abs(rendered_gap - native_gap))),
                }
        result["wrist_gaps"] = pairs
        return result

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
        self._fitting_provenance = None
