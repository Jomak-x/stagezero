"""Actual pose arrays versus StageZero scene object box proxies."""

import unittest

import numpy as np

from interaction_scene_collision import CORE27_JOINT_NAMES, scene_collision
from motion_quality import G1_JOINT_NAMES
from scene_objects import make_object


def _scene(*objects, version=2):
    doc = {"version": version, "name": "Collision test", "objects": list(objects),
           "effects": [], "lighting": "neutral"}
    if version == 3:
        doc["assets"] = [{"id": "gate-asset", "name": "Gate", "parts": [
            {"shape": "box", "position": [0, .3, 0], "size": [.8, .1, .2],
             "color": [120, 120, 120]}]}]
    return doc


def _poses(skeleton, roots):
    joints = len(CORE27_JOINT_NAMES if skeleton == "core27" else G1_JOINT_NAMES)
    result = np.zeros((len(roots), joints, 3), dtype=np.float64)
    for frame, (x, y, z) in enumerate(roots):
        result[frame, :, :] = [x, y, z]
    return result


class SceneCollisionTests(unittest.TestCase):
    def test_walk_past_wall_then_through_wall(self):
        wall = make_object("wall", 0)
        wall["position"] = [0, 1, 2]
        wall["size"] = [3, 2, .3]
        positions = _poses("core27", [(0, 1, 0), (0, 1, 2), (0, 1, 4)])
        result = scene_collision(positions, "core27", _scene(wall))
        self.assertEqual(result["total_collision_frames"], 1)
        self.assertEqual(result["per_object"][0]["first_collision_frame"], 1)
        self.assertLess(result["per_object"][0]["minimum_clearance_m"], 0)
        self.assertFalse(result["assumptions"]["mesh_or_physics_collision_verified"])

    def test_hand_collision_detected_when_root_is_clear(self):
        wall = make_object("wall", 0)
        wall["position"] = [0, 1, 2]
        wall["size"] = [3, 2, .3]
        positions = _poses("g1", [(0, 1, 0)])
        positions[0, G1_JOINT_NAMES.index("right_hand_roll_skel")] = [0, 1, 2]
        result = scene_collision(positions, "g1", _scene(wall))
        self.assertEqual(result["total_collision_frames"], 1)

    def test_procedural_arch_opening_and_post(self):
        arch = make_object("arch", 0)
        arch["position"] = [0, 1.5, 2]
        arch["size"] = [2.8, 3, .4]
        positions = _poses("g1", [(0, 1, 2), (1.078, 1, 2)])
        result = scene_collision(positions, "g1", _scene(arch))
        self.assertEqual(result["per_object"][0]["collision_frames"], 1)
        self.assertEqual(result["per_object"][0]["first_collision_frame"], 1)

    def test_rotated_custom_gate_jamb_and_opening(self):
        gate = make_object("arch", 0)
        gate.update(kind="custom", id="custom-gate", name="Custom gate",
                    asset="gate-asset", yaw=90, position=[1, 1.5, 2], size=[3, 3, .5])
        affordances = {gate["id"]: {"kind": "passage", "verified_open": True,
                                    "width_m": 2.2, "height_m": 2.4,
                                    "floor_y_m": 0, "depth_m": .5}}
        positions = _poses("core27", [(1, 1, 2), (1, 1, .8)])
        result = scene_collision(positions, "core27", _scene(gate, version=3), affordances)
        self.assertEqual(result["total_collision_frames"], 1)
        self.assertEqual(result["per_object"][0]["first_collision_frame"], 1)

    def test_custom_gate_without_verified_hole_is_solid(self):
        gate = make_object("arch", 0)
        gate.update(kind="custom", id="custom-gate", name="Custom gate",
                    asset="gate-asset", yaw=90, position=[1, 1.5, 2], size=[3, 3, .5])
        positions = _poses("core27", [(1, 1, 2)])
        result = scene_collision(positions, "core27", _scene(gate, version=3))
        self.assertEqual(result["total_collision_frames"], 1)

    def test_non_grounded_platform_is_obstacle(self):
        platform = make_object("platform", 0)
        platform["position"] = [0, .5, 2]
        platform["size"] = [2, 1, .7]
        result = scene_collision(_poses("core27", [(0, 1, 2)]), "core27", _scene(platform))
        self.assertEqual(result["total_collision_frames"], 1)

    def test_invalid_pose_and_skeleton_rejected(self):
        with self.assertRaisesRegex(ValueError, "skeleton"):
            scene_collision(_poses("core27", [(0, 1, 0)]), "bad", _scene())
        bad = _poses("g1", [(0, 1, 0)])
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            scene_collision(bad, "g1", _scene())


if __name__ == "__main__":
    unittest.main()
