"""A close limb must reject a duet window even when separated roots pass.

All generated poses below are deliberate CPU fixtures, not animation evidence.
"""
import time
import unittest

import numpy as np

from core_choreography import choreography_preset
from core_pose_cues import _library
from realtime_backend import validate_job
from realtime_clip import CanonicalClip
from studio_core_session import CoreStudioSession

IDS = ("actor_1", "actor_2")
PLACEMENTS = {IDS[0]: {"position_xz": [-1.2, 0.], "yaw": 0.},
              IDS[1]: {"position_xz": [1.2, 0.], "yaw": 0.}}


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.005)
    raise AssertionError("Fixture generation did not finish")


class LimbClient:
    def __init__(self):
        self.close_limb = False
        self.calls = []

    def wait(self, body, *, cancelled):
        validate_job(body)
        self.calls.append(body)
        source = _library()[0]["victory"]
        poses, rotations = [], []
        for actor in IDS:
            p = np.repeat(source.positions[0, :1], 40, axis=0).copy()
            p[..., 0] += PLACEMENTS[actor]["position_xz"][0] - p[0, 0, 0]
            p[..., 2] -= p[0, 0, 2]
            poses.append(p)
            rotations.append(np.repeat(source.rotations[0, :1], 40, axis=0))
        poses = np.asarray(poses)
        if self.close_limb:
            # One hand approaches the other actor's hand at a single frame.
            # Roots remain 2.4 m apart, isolating the new all-joint gate.
            poses[0, 20, 10] = poses[1, 20, 10] + [.10, 0., 0.]
        native = np.full((2, 40, 330), len(self.calls), dtype=np.float32)
        return [CanonicalClip(poses, np.asarray(rotations), 20, IDS, "ardy_core", {}, native)]


class PoseDuetSessionTests(unittest.TestCase):
    def test_limb_clearance_rejects_window_preserves_prefix_and_roundtrips_pending_recipe(self):
        client = LimbClient()
        with CoreStudioSession(client) as session:
            session.start(2, placements=PLACEMENTS)
            session.direct({a: "Stand" for a in IDS}, seconds=2)
            wait_for(lambda: session.snapshot()["total_frames"] == 40)
            prefix = session.timeline_clip()
            client.close_limb = True
            session.choreograph(choreography_preset("pose_duet"))
            wait_for(lambda: session.snapshot()["failure"] is not None)
            self.assertIn("joint clearance below 0.15", session.snapshot()["failure"])
            self.assertEqual(session.snapshot()["total_frames"], 40)
            np.testing.assert_array_equal(session.timeline_clip().positions, prefix.positions)
            np.testing.assert_array_equal(session.timeline_clip().native_features, prefix.native_features)
            content = session.save()
            client.close_limb = False
        # Loading exact output must not start inference. Pending profile metadata
        # survives and can be explicitly retried with the same native prefix.
        recovered = LimbClient()
        with CoreStudioSession(recovered) as loaded:
            loaded.load(content)
            self.assertEqual(recovered.calls, [])
            np.testing.assert_array_equal(loaded.timeline_clip().native_features, prefix.native_features)
            self.assertTrue(loaded.retry())
            def advance():
                state = loaded.snapshot()
                if state["total_frames"]: loaded.seek(state["total_frames"] - 1)
                return loaded.snapshot()["queued_stages"] == 0
            wait_for(advance)
            self.assertIsNone(loaded.snapshot()["failure"])
            self.assertEqual(loaded.snapshot()["total_frames"], 200)
            np.testing.assert_array_equal(loaded.timeline_clip().native_features[:, :40], prefix.native_features)
            self.assertEqual([b["seed"] for b in recovered.calls], [6201, 6202, 6203, 6204])
            self.assertTrue(all(np.asarray(b["history"]["native_features"]).shape[1] == 4 for b in recovered.calls))
            self.assertEqual([b["stage_kind"] for b in recovered.calls], ["continuation", "transition", "transition", "transition"])


if __name__ == '__main__': unittest.main()
