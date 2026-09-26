"""Behavior checks for the native timeline adapter without a browser server."""

import threading
import unittest
from types import SimpleNamespace

from studio_timeline import StudioTimeline


class FakeTimeline:
    def __init__(self):
        self.calls = []
        self.on_scrub = None
        self.end_frame = 0
        self.current_frame = 0

    def __getattr__(self, method):
        def call(*args, **kwargs):
            self.calls.append((method, args, kwargs))
            if method == "on_frame_change":
                self.on_scrub = args[0]
            elif method == "set_zoom_settings":
                self.end_frame = max(self.end_frame, kwargs["max_frames_zoom"])
            elif method == "set_frame_range":
                self.end_frame = args[1]
            elif method == "set_current_frame":
                self.current_frame = args[0]
        return call


class FakeSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.kind = "generated"
        self.positions = [0] * 100
        self.fps = 25
        self.frame = 0
        self.active_take = "one"
        self.takes = {"one": SimpleNamespace(segments=[
            {"start": 0, "end": 50, "prompt": "Wave"},
            {"start": 50, "end": 100, "prompt": "Turn"},
        ])}
        self.playing = True
        self.busy = False
        self.character_motion_enabled = True

    def seek(self, frame):
        self.frame = max(0, min(frame, len(self.positions) - 1))
        self.playing = False


class StudioTimelineTest(unittest.TestCase):
    def setUp(self):
        self.timeline = FakeTimeline()
        self.session = FakeSession()
        self.adapter = StudioTimeline(SimpleNamespace(timeline=self.timeline), self.session)

    def test_segments_are_read_only_and_scrubbing_pauses_at_clamped_frame(self):
        names = [call[0] for call in self.timeline.calls]
        self.assertLess(names.index("disable_constraints"), names.index("set_visible"))
        prompts = [call for call in self.timeline.calls if call[0] == "add_prompt"]
        self.assertEqual([(p[1][0], p[1][1:3]) for p in prompts],
                         [("Wave", (0, 50)), ("Turn", (50, 100))])
        self.timeline.on_scrub(300)
        self.assertEqual(self.session.frame, 99)
        self.assertFalse(self.session.playing)
        self.adapter.update()
        self.assertEqual([c[1][0] for c in self.timeline.calls if c[0] == "set_current_frame"][-1], 99)
        calls = len([c for c in self.timeline.calls if c[0] == "set_current_frame"])
        self.timeline.on_scrub(300)
        self.adapter.update()
        self.assertEqual(len([c for c in self.timeline.calls if c[0] == "set_current_frame"]), calls + 1)

    def test_playhead_updates_do_not_rebuild_prompt_blocks(self):
        before = len([c for c in self.timeline.calls if c[0] == "clear_prompts"])
        self.session.frame = 10
        self.adapter.update()
        self.assertEqual(len([c for c in self.timeline.calls if c[0] == "clear_prompts"]), before)
        self.assertEqual([c[1][0] for c in self.timeline.calls if c[0] == "set_current_frame"][-1], 10)

    def test_action_ids_address_the_current_take_and_disable_while_busy(self):
        timeline = FakeTimeline()
        adapter = StudioTimeline(SimpleNamespace(timeline=timeline), self.session, "command-control")
        prompts = [c for c in timeline.calls if c[0] == "add_prompt"]
        self.assertEqual([c[2]["uuid"] for c in prompts], [
            "stagezero|one|0|command-control", "stagezero|one|1|command-control",
        ])

        self.session.busy = True
        adapter.update()
        prompts = [c for c in timeline.calls if c[0] == "add_prompt"][-2:]
        self.assertEqual([c[2]["uuid"] for c in prompts], [
            "stagezero|one|0|", "stagezero|one|1|",
        ])
        self.session.busy = False
        adapter.update()
        prompts = [c for c in timeline.calls if c[0] == "add_prompt"][-2:]
        self.assertEqual(prompts[0][2]["uuid"], "stagezero|one|0|command-control")

    def test_legacy_call_keeps_legacy_prompt_ids(self):
        prompts = [c for c in self.timeline.calls if c[0] == "add_prompt"]
        self.assertEqual([c[2]["uuid"] for c in prompts], [
            "stagezero-segment-0", "stagezero-segment-1",
        ])

    def test_static_preview_disables_actions_and_preserves_playhead_until_restored(self):
        timeline = FakeTimeline()
        adapter = StudioTimeline(SimpleNamespace(timeline=timeline), self.session, "command-control")
        self.session.frame = 25
        self.session.playing = False
        self.session.character_motion_enabled = False
        adapter.update()
        prompts = [c for c in timeline.calls if c[0] == "add_prompt"][-2:]
        self.assertEqual([c[2]["uuid"] for c in prompts], [
            "stagezero|one|0|", "stagezero|one|1|",
        ])
        # Viser publishes an optimistic ruler position before its callback.
        timeline.current_frame = 80
        timeline.on_scrub(80)
        self.assertEqual(self.session.frame, 25)
        self.assertEqual(self.session.active_take, "one")
        adapter.update()
        self.assertEqual(timeline.current_frame, 25)
        self.session.character_motion_enabled = True
        adapter.update()
        prompts = [c for c in timeline.calls if c[0] == "add_prompt"][-2:]
        self.assertEqual(prompts[0][2]["uuid"], "stagezero|one|0|command-control")
        timeline.on_scrub(80)
        self.assertEqual(self.session.frame, 80)

    def test_ruler_fits_the_exact_take_length(self):
        zoom = [c for c in self.timeline.calls if c[0] == "set_zoom_settings"][-1]
        self.assertEqual(zoom[2], {"default_num_frames_zoom": 100, "max_frames_zoom": 100})
        frame_range = [c for c in self.timeline.calls if c[0] == "set_frame_range"][-1]
        self.assertEqual(frame_range[1], (0, 99))
        self.assertEqual(self.timeline.end_frame, 99)

        self.session.positions = [0]
        self.adapter.update()
        frame_range = [c for c in self.timeline.calls if c[0] == "set_frame_range"][-1]
        self.assertEqual(frame_range[1], (0, 0))
        self.assertEqual(self.timeline.end_frame, 0)
        zoom = [c for c in self.timeline.calls if c[0] == "set_zoom_settings"][-1]
        self.assertEqual(zoom[2], {"default_num_frames_zoom": 1, "max_frames_zoom": 1})

    def test_scrubbing_does_not_cancel_generation(self):
        self.session.busy = True
        self.timeline.current_frame = 40
        self.timeline.on_scrub(40)
        self.assertEqual(self.session.frame, 0)
        self.assertTrue(self.session.busy)
        self.adapter.update()
        self.assertEqual(self.timeline.current_frame, 0)

    def test_reference_pose_hides_timeline_and_ignores_scrubs(self):
        self.session.kind = "reference"
        self.adapter.update()
        self.assertEqual([c[1][0] for c in self.timeline.calls if c[0] == "set_visible"][-1], False)
        self.timeline.on_scrub(40)
        self.assertEqual(self.session.frame, 0)


if __name__ == "__main__":
    unittest.main()

class CoreTimelineTests(unittest.TestCase):
    def test_native_timeline_uses_twenty_fps_and_routes_scrub_without_touching_g1(self):
        class Core:
            def __init__(self):
                self.state = dict(active=True,total_frames=80,frame=7,
                    segments=[{'start':0,'end':40,'prompt':'Two people gesture'},
                              {'start':40,'end':80,'prompt':'They turn'}])
            def snapshot(self): return dict(self.state)
            def seek(self, frame): self.state['frame']=frame
        core=Core(); timeline=FakeTimeline(); g1=FakeSession()
        adapter=StudioTimeline(SimpleNamespace(timeline=timeline),g1,core_session=core)
        self.assertIn(('set_fps',(20.0,),{}),timeline.calls)
        self.assertEqual(timeline.end_frame,79)
        timeline.on_scrub(999)
        self.assertEqual(core.state['frame'],79)
        self.assertEqual(g1.frame,0)
        core.state['active']=False
        adapter.update()
        self.assertEqual(timeline.end_frame,99)
        self.assertIn(('set_fps',(25.0,),{}),timeline.calls)
        timeline.on_scrub(21)
        self.assertEqual(g1.frame,21)

    def test_paired_research_timeline_has_separate_source_and_scrub(self):
        class Motion:
            def __init__(self, active, prompt):
                self.state = dict(active=active, total_frames=120, frame=8,
                                  segments=[{'start': 0, 'end': 120, 'prompt': prompt}])

            def snapshot(self):
                return dict(self.state)

            def seek(self, frame):
                self.state['frame'] = frame

        core, paired = Motion(False, 'Native Core'), Motion(True, 'Joint research pair')
        ruler, g1 = FakeTimeline(), FakeSession()
        adapter = StudioTimeline(SimpleNamespace(timeline=ruler), g1,
                                 core_session=core, paired_session=paired)
        prompts = [call for call in ruler.calls if call[0] == 'add_prompt']
        self.assertEqual(prompts[-1][1][0], 'Joint research pair')
        self.assertEqual(prompts[-1][2]['uuid'], 'paired-research-scene-0')
        ruler.on_scrub(999)
        self.assertEqual(paired.state['frame'], 119)
        self.assertEqual(g1.frame, 0)
        paired.state['active'] = False
        core.state['active'] = True
        adapter.update()
        self.assertEqual([call for call in ruler.calls if call[0] == 'add_prompt'][-1][1][0], 'Native Core')

    def test_native_playhead_does_not_republish_timeline_layout(self):
        state=dict(active=True,total_frames=40,frame=0,segments=[])
        core=SimpleNamespace(snapshot=lambda:dict(state),seek=lambda f:None)
        timeline=FakeTimeline()
        adapter=StudioTimeline(SimpleNamespace(timeline=timeline),FakeSession(),core_session=core)
        count=sum(c[0]=='clear_prompts' for c in timeline.calls)
        state['frame']=17;adapter.update()
        self.assertEqual(sum(c[0]=='clear_prompts' for c in timeline.calls),count)
        self.assertEqual(timeline.current_frame,17)


class NativePairFrameRateTests(unittest.TestCase):
    def test_native_pair_uses_30fps_and_refreshes_on_rate_change(self):
        state = dict(active=True, total_frames=210, frame=30, fps=30, segments=[])
        pair = SimpleNamespace(snapshot=lambda: dict(state), seek=lambda f: None)
        ruler = FakeTimeline()
        adapter = StudioTimeline(SimpleNamespace(timeline=ruler), FakeSession(), paired_session=pair)
        self.assertIn(('set_fps', (30.0,), {}), ruler.calls)
        state['fps'] = 20
        adapter.update()
        self.assertEqual([c for c in ruler.calls if c[0] == 'set_fps'][-1][1], (20.0,))
