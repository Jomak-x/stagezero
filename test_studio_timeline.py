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

    def __getattr__(self, method):
        def call(*args, **kwargs):
            self.calls.append((method, args, kwargs))
            if method == "on_frame_change":
                self.on_scrub = args[0]
            elif method == "set_zoom_settings":
                self.end_frame = max(self.end_frame, kwargs["max_frames_zoom"])
            elif method == "set_frame_range":
                self.end_frame = args[1]
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
        self.timeline.on_scrub(40)
        self.assertEqual(self.session.frame, 0)
        self.assertTrue(self.session.busy)

    def test_reference_pose_hides_timeline_and_ignores_scrubs(self):
        self.session.kind = "reference"
        self.adapter.update()
        self.assertEqual([c[1][0] for c in self.timeline.calls if c[0] == "set_visible"][-1], False)
        self.timeline.on_scrub(40)
        self.assertEqual(self.session.frame, 0)


if __name__ == "__main__":
    unittest.main()
