"""The Together panel previews intent before it changes Core motion."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import math
import threading
import time
from unittest import TestCase
from unittest.mock import patch

from core_choreography import choreography_preset
from studio_core_controls import CoreStudioControls
from studio_interaction_scene import recommend_placements
from test_studio_core_controls import Core, Gui, Studio


class ChoreographyCore(Core):
    def reset(self, *, actor_count, scene_document, placements):
        self.last_placements = placements
        return super().start(actor_count=actor_count, scene_document=scene_document,
                             placements=placements)

    def choreograph(self, plan):
        self.calls.append(("choreograph", plan))


class PairedResearch:
    def __init__(self):
        self.active = False
        self.busy = False
        self.frames = 0
        self.calls = []

    def snapshot(self):
        return {"active": self.active, "available": True, "busy": self.busy,
                "total_frames": self.frames, "status": "Paired research ready"}

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def generate(self, prompt, seed, *, frames, scene_document):
        self.calls.append(("generate", prompt, seed, frames, scene_document))
        self.busy = True

    def cancel(self):
        self.calls.append(("cancel",))
        self.busy = False

    def save(self):
        self.calls.append(("save",))
        return b"paired-research-archive"

    def load(self, content):
        self.calls.append(("load", content))
        self.active = True
        self.frames = 120

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))

    def restart(self):
        self.calls.append(("restart",))


class TogetherControlsTests(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.core = ChoreographyCore()
        self.gui = Gui()
        self.controls = CoreStudioControls(
            self.gui, self.core, Studio(), on_active=lambda _: None,
            project_folder=Path(temp.name))

    def _start_two(self):
        self.controls.enabled.edit(True, client=object())
        self.controls.cast.value = "Two actors"
        self.controls.start.click()

    def test_two_actors_required_and_generation_is_explicit(self):
        self.assertEqual(self.controls.together_preset.value, "Pose duet")
        self.controls.enabled.edit(True, client=object())
        self.assertTrue(self.controls.together_preview_preset.disabled)
        self.assertTrue(self.controls.together_generate.disabled)
        self._start_two()
        self.controls.together_preset.value = "Feint and dodge"
        self.assertFalse(self.controls.together_preview_preset.disabled)
        self.controls.together_preview_preset.click()
        self.assertFalse(any(call[0] == "choreograph" for call in self.core.calls))
        self.assertFalse(self.controls.together_generate.disabled)
        self.assertIn("Ready together", self.controls.together_preview.content)
        self.assertIn("Motion has not been generated", self.controls.together_preview.content)
        self.controls.together_generate.click()
        submitted = self.core.calls[-1]
        self.assertEqual(submitted[0], "choreograph")
        self.assertEqual(submitted[1]["name"], "feint_dodge")

    def test_new_performance_backs_up_motion_and_faces_partners(self):
        self.controls.enabled.edit(True, client=object())
        self.core.total_frames = 40
        self.controls.tick()
        self.controls.together_start.click()
        self.assertEqual(self.core.calls[-2][0], "save")
        self.assertEqual(self.core.calls[-1][:2], ("start", 2))
        points = self.core.last_placements
        a, b = points["actor_1"], points["actor_2"]
        self.assertEqual(a["position_xz"], [-2.25, 0.0])
        self.assertEqual(b["position_xz"], [0.0, 0.0])
        self.assertLessEqual(a["position_xz"][0], b["position_xz"][0])
        dx = b["position_xz"][0] - a["position_xz"][0]
        dz = b["position_xz"][1] - a["position_xz"][1]
        self.assertGreater(math.hypot(dx, dz), 1.4)
        self.assertGreaterEqual(math.hypot(dx, dz), 2.25)
        self.assertAlmostEqual(a["yaw"], math.atan2(dx, dz))
        self.assertAlmostEqual(b["yaw"], math.atan2(-dx, -dz))
        self.assertFalse(self.controls.together_start.disabled)

    def test_wider_together_gap_preserves_old_placement_default(self):
        document = Studio().scene_document()
        default = recommend_placements(document, 2)
        wider = recommend_placements(document, 2, minimum_separation_m=2.0)
        self.assertEqual(default, recommend_placements(document, 2, minimum_separation_m=1.5))
        self.assertEqual(default[0]["position_xz"], wider[0]["position_xz"])
        self.assertGreaterEqual(math.dist(wider[0]["position_xz"], wider[1]["position_xz"]), 2.0)
        for invalid in (.64, 5.01, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                recommend_placements(document, 2, minimum_separation_m=invalid)

    def test_new_pose_duet_cast_faces_partners_at_safe_gap(self):
        self.controls.together_preset.value = "Pose duet"
        self.controls.together_start.click()
        a, b = (self.core.last_placements[aid] for aid in ("actor_1", "actor_2"))
        dx = b["position_xz"][0] - a["position_xz"][0]
        dz = b["position_xz"][1] - a["position_xz"][1]
        self.assertGreaterEqual(math.hypot(dx, dz), 2.25)
        self.assertAlmostEqual(a["yaw"], math.atan2(dx, dz))
        self.assertAlmostEqual(b["yaw"], math.atan2(-dx, -dz))

    def test_existing_two_person_cast_resets_only_after_explicit_new_performance_click(self):
        self._start_two()
        self.core.total_frames = 40
        self.controls.tick()
        previous_calls = len(self.core.calls)
        self.assertEqual(len(self.core.calls), previous_calls)
        self.controls.together_start.click()
        self.assertEqual(self.core.calls[-2][0], "save")
        self.assertEqual(self.core.calls[-1][:2], ("start", 2))
        self.assertGreaterEqual(math.dist(*[self.core.last_placements[aid]["position_xz"]
                                           for aid in ("actor_1", "actor_2")]), 2.25)

    def test_swap_roles_and_invalidate_preview_on_change(self):
        self._start_two()
        self.controls.together_preview_preset.click()
        original = self.controls._together_plan
        self.controls.together_swap.edit(True, client=object())
        self.assertTrue(self.controls.together_generate.disabled)
        self.controls.together_preview_preset.click()
        swapped = self.controls._together_plan
        self.assertEqual(swapped["beats"][1]["actor_prompts"]["actor_1"],
                         original["beats"][1]["actor_prompts"]["actor_2"])
        self.controls.together_preset.edit("Dance and answer", client=object())
        self.assertTrue(self.controls.together_generate.disabled)

    def test_preview_uses_valid_literal_mdx_for_model_text(self):
        self._start_two()
        plan = choreography_preset("feint_dodge")
        plan["name"] = "Scene {one} <br> *sparring*"
        plan["beats"][0]["name"] = "Ready {together}"
        plan["beats"][0]["actor_prompts"]["actor_1"] = "Look {left} <br> and [turn](now)."
        self.controls._set_together_plan(plan, self.controls._together_state())
        self.controls.tick()
        preview = self.controls.together_preview.content
        self.assertIn("&#123;one&#125;", preview)
        self.assertIn("&lt;br&gt;", preview)
        self.assertIn(r"\*sparring\*", preview)
        self.assertIn("\n\n**Actor 1:**", preview)
        self.assertIn("\n\n**Actor 2:**", preview)
        self.assertNotIn("<br>", preview)
        self.assertNotIn("{", preview)
        self.assertFalse(self.controls.together_generate.disabled)
        self.controls._together_plan = None
        self.controls._together_message = "AI rejected {invalid} <response>"
        self.controls.tick()
        self.assertIn("&#123;invalid&#125;", self.controls.together_preview.content)
        self.assertIn("&lt;response&gt;", self.controls.together_preview.content)

    def test_ai_runs_off_ui_thread_and_ignores_changed_direction(self):
        self._start_two()
        entered, release = threading.Event(), threading.Event()

        class Planner:
            @classmethod
            def from_env(cls):
                return cls()

            def generate(self, intent, ids, *, seed):
                entered.set()
                release.wait(2)
                return choreography_preset("dance_response", ids, seed=seed)

        with patch("core_choreography_ai.ChoreographyPlanner", Planner):
            self.controls.together_direction.edit("Dance together", client=object())
            start = time.monotonic()
            self.controls.together_plan_ai.click()
            self.assertLess(time.monotonic() - start, .5)
            self.assertTrue(entered.wait(1))
            self.assertTrue(self.controls.together_generate.disabled)
            self.controls.together_direction.edit("Different scene", client=object())
            release.set()
            for _ in range(40):
                self.controls.tick()
                if not self.controls._ai_pending:
                    break
                time.sleep(.01)
            self.assertIsNone(self.controls._together_plan)
            self.assertTrue(self.controls.together_generate.disabled)
            self.assertIn("ignored", self.controls.together_preview.content)

    def test_ai_candidate_requires_separate_generation_click(self):
        self._start_two()

        class Planner:
            @classmethod
            def from_env(cls):
                return cls()

            def generate(self, intent, ids, *, seed):
                return choreography_preset("surprise_celebration", ids, seed=seed)

        with patch("core_choreography_ai.ChoreographyPlanner", Planner):
            self.controls.together_direction.edit("A surprise", client=object())
            self.controls.together_plan_ai.click()
            for _ in range(100):
                self.controls.tick()
                if self.controls._together_plan is not None:
                    break
                time.sleep(.01)
            self.assertIsNotNone(self.controls._together_plan)
            self.assertFalse(any(call[0] == "choreograph" for call in self.core.calls))
            self.assertFalse(self.controls.together_generate.disabled)
            self.controls.together_generate.click()
            self.assertEqual(self.core.calls[-1][0], "choreograph")

    def test_changed_direction_cannot_start_overlapping_ai_request(self):
        self._start_two()
        entered, release = threading.Event(), threading.Event()
        calls = []

        class Planner:
            @classmethod
            def from_env(cls):
                return cls()

            def generate(self, intent, ids, *, seed):
                calls.append(intent)
                entered.set()
                if intent == "First idea":
                    release.wait(2)
                return choreography_preset("dance_response", ids, seed=seed)

        with patch("core_choreography_ai.ChoreographyPlanner", Planner):
            self.controls.together_direction.edit("First idea", client=object())
            self.controls.together_plan_ai.click()
            self.assertTrue(entered.wait(1))
            self.controls.together_direction.edit("Second idea", client=object())
            self.assertTrue(self.controls.together_plan_ai.disabled)
            self.assertTrue(self.controls._ai_pending)
            self.controls.together_plan_ai.click()  # Guard even if a stale UI click arrives.
            self.assertEqual(calls, ["First idea"])
            release.set()
            for _ in range(100):
                self.controls.tick()
                if not self.controls._ai_pending:
                    break
                time.sleep(.01)
            self.assertIsNone(self.controls._together_plan)
            self.assertFalse(self.controls.together_plan_ai.disabled)
            self.controls.together_plan_ai.click()
            for _ in range(100):
                self.controls.tick()
                if self.controls._together_plan is not None:
                    break
                time.sleep(.01)
            self.assertEqual(calls, ["First idea", "Second idea"])


class PairedResearchControlsTests(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.core = ChoreographyCore()
        self.paired = PairedResearch()
        self.gui = Gui()
        self.switches = []

        def switch(active):
            self.switches.append(active)
            self.paired.activate() if active else self.paired.deactivate()

        self.controls = CoreStudioControls(
            self.gui, self.core, Studio(), on_active=lambda _: None,
            project_folder=Path(temp.name) / "core-projects",
            paired_session=self.paired, on_paired_active=switch)

    def test_joint_pair_example_and_generation_do_not_touch_native_core(self):
        self.assertEqual(self.controls.pair_prompt.value,
                         "Two people perform a choreographed martial arts exchange: sidestep dodge, forearm block, controlled push, then step apart.")
        self.controls.pair_idea.value = "Partner dance"
        self.controls.pair_example.click()
        self.assertIn("hold hands and dance", self.controls.pair_prompt.value)
        self.assertEqual(self.controls.pair_seed.value, "7302")
        self.controls.pair_generate.click()
        self.assertEqual(self.switches, [True])
        action = self.paired.calls[-1]
        self.assertEqual(action[:4], ("generate", self.controls.pair_prompt.value, 7302, 120))
        self.assertEqual(self.core.calls, [])
        self.assertTrue(self.controls.pair_generate.disabled)
        self.assertFalse(self.controls.pair_cancel.disabled)
        self.controls.pair_cancel.click()
        self.assertEqual(self.paired.calls[-1], ("cancel",))

    def test_paired_archive_is_separate_and_replay_switches_modes(self):
        self.paired.frames = 120
        self.paired.active = True
        self.controls.tick()
        self.controls.pair_save.click()
        paths = list(self.controls.pair_folder.glob("*.paired.stagezero.npz"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].read_bytes(), b"paired-research-archive")
        self.assertEqual(list(self.controls.folder.glob("*.core.stagezero.npz")), [])
        self.paired.active = False
        self.controls.pair_open.click()
        self.assertEqual(self.paired.calls[-1], ("load", b"paired-research-archive"))
        self.assertEqual(self.switches[-1], True)
        self.controls.pair_play.click()
        self.assertEqual(self.paired.calls[-1], ("play",))
        self.controls.pair_back.click()
        self.assertEqual(self.switches[-1], False)
