"""Native conditioning schedule for turn-then-walk relative commands."""
import math
import unittest

import numpy as np

from realtime_backend import validate_job
from realtime_client import request_body
from realtime_director import RealtimeDirector
from realtime_navigation import plan_navigation
from scene_objects import make_object
from test_realtime_navigation import scene


class RelativeTurnTests(unittest.TestCase):
    def plan(self, goal, yaw=0., *, witness=False):
        ids = ("walker", "witness") if witness else ("walker",)
        placements = {"walker": {"position_xz": [0., 0.], "yaw": yaw}}
        if witness:
            placements["witness"] = {"position_xz": [4., 4.], "yaw": math.pi}
        stages, route = plan_navigation(scene(), ids, actor_id="walker", verb="move",
                                        target_xz=goal, initial_placements=placements)
        director = RealtimeDirector(ids)
        director.queue_sequence(stages)
        validate_job(request_body(director.claim_request()))
        return stages, route

    def test_left_right_and_half_turn_have_stationary_eased_native_prefix(self):
        for goal, sign, direction in (([-1., 0.], -1, "left"), ([1., 0.], 1, "right"), ([0., -1.], 1, "right")):
            with self.subTest(goal=goal):
                stages, route = self.plan(goal, witness=True)
                self.assertEqual(route["schedule"]["initial_turn_frames"], 40)
                self.assertEqual(route["schedule"]["frames"], sum(stage.frames for stage in stages))
                self.assertEqual(route["schedule"]["movement_start_frame"], 40)
                self.assertEqual([s.metadata["navigation"]["stage"] for s in stages], list(range(len(stages))))
                turn = stages[0]
                self.assertEqual(turn.prompt, f"A person turns {direction} in place, keeping an upright posture.")
                targets = turn.metadata["root_targets"]["walker"]
                self.assertTrue(all(t["position_xz"] == [0., 0.] for t in targets))
                self.assertEqual(targets[0]["frame"], 0)
                self.assertEqual(targets[0]["heading"], 0.)
                headings = np.unwrap([t["heading"] for t in targets])
                self.assertTrue(np.all(sign*np.diff(headings) > 0))
                for target in targets:
                    fraction = target["frame"]/39
                    expected = route["schedule"]["initial_turn_radians"] * fraction*fraction*(3-2*fraction)
                    self.assertAlmostEqual(target["heading"], expected, places=5)
                self.assertTrue(all(t["position_xz"] == [4., 4.] and t["heading"] == math.pi for t in turn.metadata["root_targets"]["witness"]))
                self.assertEqual(stages[1].prompt, "A person walks forward naturally.")
                self.assertEqual(stages[-1].prompt, "A person stands upright and relaxed.")
                self.assertTrue(all("turn" not in s.prompt for s in stages[1:]))
                self.assertNotIn("initial_placements", stages[1].metadata)
                np.testing.assert_allclose(stages[1].metadata["root_targets"]["walker"][0]["position_xz"], np.asarray(goal)*.26, atol=1e-5)

    def test_wrap_takes_shortest_turn_and_keeps_headings_in_backend_range(self):
        target_heading = math.radians(-150)
        stages, route = self.plan([math.sin(target_heading), math.cos(target_heading)], math.radians(170))
        self.assertAlmostEqual(route["schedule"]["initial_turn_radians"], math.radians(40), places=5)
        targets = stages[0].metadata["root_targets"]["walker"]
        headings = [t["heading"] for t in targets]
        self.assertTrue(all(-math.pi <= heading <= math.pi for heading in headings))
        self.assertTrue(np.all(np.diff(np.unwrap(headings)) > 0))
        self.assertAlmostEqual(headings[-1], target_heading, places=5)
        self.assertIn("turns right", stages[0].prompt)

    def test_straight_and_small_heading_changes_do_not_add_turn_horizon(self):
        for angle in (0., math.radians(20)):
            stages, route = self.plan([math.sin(angle), math.cos(angle)])
            self.assertEqual(route["schedule"]["initial_turn_frames"], 0)
            self.assertEqual(len(stages), 2)
            self.assertEqual(stages[0].prompt, "A person walks forward naturally.")
            self.assertEqual(stages[-1].metadata["navigation"]["phase"], "hold")

    def test_prefix_counts_toward_existing_thirty_second_cap(self):
        _, route = self.plan([0., 17.])
        self.assertEqual(route["schedule"]["frames"], 600)
        with self.assertRaisesRegex(ValueError, "30-second"):
            self.plan([17., 0.])

    def test_object_navigation_keeps_its_original_schedule_prompts_and_targets(self):
        box = make_object("crate", 0)
        box.update(id="box", name="Box", position=[0., .4, 3.], size=[.4, .8, .4])
        stages, route = plan_navigation(scene(box), ("walker",), actor_id="walker", target_id="box",
                                       verb="approach", initial_placements={"walker": {"position_xz": [0., 0.], "yaw": math.pi/2}})
        self.assertEqual(len(stages), 3)
        self.assertEqual(route["schedule"]["frames"], 120)
        self.assertNotIn("initial_turn_frames", route["schedule"])
        self.assertTrue(all(s.prompt == "Approach Box along the planned clear route, then stop." for s in stages))
        self.assertTrue(all("phase" not in s.metadata["navigation"] for s in stages))
        first = stages[0].metadata["root_targets"]["walker"][0]
        self.assertEqual(first, {"frame": 7, "position_xz": [0., .26], "heading": 0.})


    def test_spatial_object_route_can_opt_into_turn_without_changing_walk_targets(self):
        box = make_object("crate", 0)
        box.update(id="box", name="Box", position=[0., .4, 3.], size=[.4, .8, .4])
        kw = dict(actor_id="walker", target_id="box", verb="approach",
                  initial_placements={"walker":{"position_xz":[0.,0.],"yaw":math.pi/2}})
        normal, _ = plan_navigation(scene(box), ("walker",), **kw)
        turned, route = plan_navigation(scene(box), ("walker",), turn_before_travel=True, **kw)
        self.assertEqual(route["schedule"]["initial_turn_frames"], 40)
        self.assertIn("turns left", turned[0].prompt)
        for baseline, suffix in zip(normal, turned[1:]):
            self.assertEqual(baseline.prompt, suffix.prompt)
            self.assertEqual(baseline.metadata["root_targets"], suffix.metadata["root_targets"])

    def test_explicit_spatial_gait_turns_then_walks_without_heading_or_extra_hold(self):
        starts = {"walker": {"position_xz": [0., 0.], "yaw": 0.},
                  "witness": {"position_xz": [4., 4.], "yaw": math.pi}}
        kw = dict(actor_id="walker", verb="move", target_xz=[-2., 0.],
                  initial_placements=starts, speed_mps=1.2, turn_before_travel=True)
        stages, route = plan_navigation(scene(), ("walker", "witness"), gait_profile="spatial", **kw)
        default, _ = plan_navigation(scene(), ("walker", "witness"), **kw)
        self.assertEqual(route["schedule"]["initial_turn_frames"], 40)
        self.assertEqual(route["schedule"]["terminal_hold_frames"], 0)
        self.assertEqual(route["schedule"]["terminal_settle_frames"], 8)
        self.assertEqual(len(default), len(stages) + 1)
        self.assertEqual(stages[0].metadata["root_targets"], default[0].metadata["root_targets"])
        self.assertIn("turns left", stages[0].prompt)
        self.assertEqual(stages[-1].metadata["navigation"]["phase"], "walk")
        self.assertIn("relaxed stop", stages[-1].prompt)
        self.assertTrue(all("heading" not in goal for stage in stages[1:]
                            for goal in stage.metadata["root_targets"]["walker"]))
        self.assertEqual([s.metadata["root_targets"]["witness"] for s in stages],
                         [s.metadata["root_targets"]["witness"] for s in default[:-1]])
        self.assertEqual(stages[-1].metadata["root_targets"]["walker"][-1]["position_xz"], [-2., 0.])

    def test_default_profile_is_identical_to_omitted_profile(self):
        kw = dict(actor_id="walker", verb="move", target_xz=[0., 2.],
                  initial_placements={"walker": {"position_xz": [0., 0.], "yaw": 0.}})
        implicit, implicit_route = plan_navigation(scene(), ("walker",), **kw)
        explicit, explicit_route = plan_navigation(scene(), ("walker",), gait_profile="default", **kw)
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit_route, explicit_route)


if __name__ == "__main__":
    unittest.main()
