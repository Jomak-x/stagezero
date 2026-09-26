"""Geometry checks for scene-grounded action plans; no model or Pod needed."""

import math
import unittest

from interaction_planner import plan_action
from scene_objects import make_object


def _scene(*objects, version=2):
    data = {"version": version, "name": "Test", "objects": list(objects),
            "effects": [], "lighting": "neutral"}
    if version == 3:
        data["assets"] = [{"id": "gate-asset", "name": "Gate", "parts": [
            {"shape": "box", "position": [0, .3, 0], "size": [.8, .1, .2],
             "color": [120, 120, 120]}]}]
    return data


def _arch(identifier="gate-1", x=0, z=2, width=2.8):
    obj = make_object("arch", 1)
    obj.update(id=identifier, name="North gate", position=[x, 1.5, z], size=[width, 3., .4])
    return obj


def _custom_gate(yaw=90., width=3.):
    obj = make_object("arch", 1)
    obj.update(id="red-gate", name="Red gate", kind="custom", asset="gate-asset",
               yaw=yaw, position=[1., 1.5, 2.], size=[width, 3., .5])
    return obj


def _action(verb="go_through", **changes):
    return dict({"verb": verb, "actor_id": "actor-a", "target_id": "gate-1"}, **changes)


class ScenePlannerTests(unittest.TestCase):
    def test_procedural_arch_has_entry_center_exit_and_timing(self):
        plan = plan_action(_action(start_seconds=3), _scene(_arch()), [0, 0, 0])
        self.assertEqual([w["role"] for w in plan["waypoints"]],
                         ["start", "entry", "center", "exit"])
        self.assertEqual(plan["waypoints"][2]["position_xz"], [0., 2.])
        self.assertGreater(plan["waypoints"][-1]["time_seconds"], 3.)
        self.assertEqual(plan["geometry"]["source"], "procedural_arch")
        self.assertFalse(plan["assumptions"]["motion_following_verified"])

    def test_rotated_custom_gate_uses_viser_yaw_not_flat_bbox(self):
        gate = _custom_gate(yaw=90)
        aff = {"red-gate": {"kind": "passage", "verified_open": True,
                             "width_m": 2.2, "height_m": 2.5, "depth_m": .5, "floor_y_m": 0}}
        action = {"verb": "go_through", "actor_id": "actor-a", "target_id": "red-gate"}
        plan = plan_action(action, _scene(gate, version=3), [-1, 0, 2], affordances=aff)
        entry, center, exit_point = [w["position_xz"] for w in plan["waypoints"][-3:]]
        self.assertAlmostEqual(center[0], 1)
        self.assertAlmostEqual(center[1], 2)
        self.assertLess(entry[0], center[0])
        self.assertGreater(exit_point[0], center[0])
        self.assertAlmostEqual(entry[1], center[1])
        self.assertEqual(plan["geometry"]["yaw_degrees"], 90.)

    def test_obstacle_causes_waypoint_detour(self):
        box = make_object("crate", 0)
        box["position"] = [0., .4, .3]
        plan = plan_action(_action(), _scene(_arch(), box), [0, 0, -1])
        self.assertGreater(len(plan["waypoints"]), 4)
        self.assertTrue(any(abs(w["position_xz"][0]) > .6
                            for w in plan["waypoints"] if w["role"] == "route"))

    def test_blocked_opening_rejected_even_if_approach_possible(self):
        box = make_object("crate", 0)
        box["position"] = [0., .4, 2.]
        with self.assertRaisesRegex(ValueError, "blocked"):
            plan_action(_action(), _scene(_arch(), box), [0, 0, 0])

    def test_narrow_or_closed_passage_is_not_faked(self):
        with self.assertRaisesRegex(ValueError, "narrow"):
            plan_action(_action(), _scene(_arch(width=1.)), [0, 0, 0])
        gate = _custom_gate()
        action = {"verb": "go_through", "actor_id": "actor-a", "target_id": "red-gate"}
        with self.assertRaisesRegex(ValueError, "verified open"):
            plan_action(action, _scene(gate, version=3), [1, 0, 0])

    def test_floating_sunken_and_raised_platform_rejected(self):
        for floor in (.5, -.5):
            gate = _arch()
            gate["position"][1] += floor
            with self.subTest(floor=floor), self.assertRaisesRegex(ValueError, "not grounded"):
                plan_action(_action(), _scene(gate), [0, 0, 0])
        gate = _arch()
        platform = make_object("platform", 0)
        platform["size"] = [2., 1., .7]
        platform["position"] = [0., .5, 2.]
        with self.assertRaisesRegex(ValueError, "blocked"):
            plan_action(_action(), _scene(gate, platform), [0, 0, 0])

    def test_custom_gate_requires_explicit_grounded_floor(self):
        gate = _custom_gate()
        action = {"verb": "go_through", "actor_id": "actor-a", "target_id": "red-gate"}
        affordance = {"kind": "passage", "verified_open": True,
                      "width_m": 2., "height_m": 2.4}
        with self.assertRaisesRegex(ValueError, "floor_y_m"):
            plan_action(action, _scene(gate, version=3), [0, 0, 2], affordances={gate["id"]: affordance})
        affordance["floor_y_m"] = .5
        with self.assertRaisesRegex(ValueError, "not grounded"):
            plan_action(action, _scene(gate, version=3), [0, 0, 2], affordances={gate["id"]: affordance})

    def test_elevated_actor_start_and_low_non_target_arch(self):
        with self.assertRaisesRegex(ValueError, "ground anchor"):
            plan_action(_action(), _scene(_arch()), [0, 1.2, 0])
        low = _arch("low", z=.5)
        low["size"][1] = 1.5
        low["position"][1] = .75
        target = make_object("crate", 0)
        target["position"] = [0, .4, 2.]
        plan = plan_action({"verb": "approach", "actor_id": "a", "target_id": target["id"]},
                           _scene(low, target), [0, 0, -2])
        self.assertGreater(len(plan["waypoints"]), 2)
        self.assertTrue(any(abs(w["position_xz"][0]) > .8 for w in plan["waypoints"] if w["role"] == "route"))

    def test_ambiguous_names_and_unknown_id_rejected(self):
        first, second = _arch("a"), _arch("b", x=4)
        action = {"verb": "go_through", "actor_id": "actor-a", "target_name": "North gate"}
        with self.assertRaisesRegex(ValueError, "matched 2"):
            plan_action(action, _scene(first, second), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Unknown target_id"):
            plan_action(_action(), _scene(first), [0, 0, 0])

    def test_approach_routes_to_near_side_of_solid_object(self):
        box = make_object("crate", 0)
        box["position"] = [0, .4, 2.]
        action = {"verb": "approach", "actor_id": "actor-a", "target_id": box["id"]}
        plan = plan_action(action, _scene(box), [0, 0, 0])
        self.assertEqual(plan["waypoints"][-1]["role"], "approach")
        self.assertLess(plan["waypoints"][-1]["position_xz"][1], 2.)

    def test_invalid_units_coordinates_and_nonfinite_values(self):
        scene = _scene(_arch())
        with self.assertRaisesRegex(ValueError, "actor_position"):
            plan_action(_action(), scene, [0, 0])
        with self.assertRaisesRegex(ValueError, "finite"):
            plan_action(_action(), scene, [math.nan, 0, 0])
        scene["objects"][0]["size"] = [280, 300, 40]  # centimetre-like input
        with self.assertRaisesRegex(ValueError, "metre"):
            plan_action(_action(), scene, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
