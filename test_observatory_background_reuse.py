"""A second authored background reuses, but never regenerates, native motion."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.observatory_background_reuse import (
    prepare_observatory_artifacts, revalidate_observatory_background,
)
from terrain_assisted_session import (generate_native_terrain_commands,
                                      load_native_terrain_result,
                                      save_native_terrain_result)
from test_terrain_assisted_session import TargetClient
from traversal_kit import (TEMPLE_START_ROOT_XYZ, traversable_temple_scene,
                           traversable_observatory_scene)


class ObservatoryBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = TargetClient()
        client.root = np.asarray(TEMPLE_START_ROOT_XYZ)+[0., .02, 0.]
        cls.source = generate_native_terrain_commands(
            traversable_temple_scene(),
            "walk up Shallow temple stairs, cross Suspended temple bridge, "
            "open Temple gate, and enter",
            actor_ids=("actor_1",), actor_id="actor_1",
            initial_placements={"actor_1": {"position_xz": [0., TEMPLE_START_ROOT_XYZ[2]],
                                              "yaw": math.pi}},
            client=client)

    def test_new_scene_remeasures_identical_native_features_and_reactions(self):
        result = revalidate_observatory_background(self.source)
        np.testing.assert_array_equal(result.native_clip.positions,
                                      self.source.native_clip.positions)
        np.testing.assert_array_equal(result.native_clip.native_features,
                                      self.source.native_clip.native_features)
        self.assertEqual([route["target_id"] for route in result.routes],
                         ["observatory-stairs", "observatory-bridge",
                          "observatory-gate", "observatory-gate"])
        self.assertEqual(tuple(route["schedule"]["frames"] for route in result.routes),
                         tuple(end-start for start, end in self.source.action_spans))
        self.assertTrue(all(row["completed"] for row in result.measurements))
        self.assertTrue(result.measurements[-1]["crossing_verified"])
        self.assertEqual(next(row for row in result.reaction_states
                              if row["id"] == "observatory-gate")["opening_fraction"], 1.)

    def test_missing_alternate_bridge_fails_instead_of_reusing_old_route(self):
        scene = traversable_observatory_scene()
        scene["objects"] = [obj for obj in scene["objects"]
                            if obj["id"] != "observatory-bridge"]
        with patch("experiments.observatory_background_reuse.traversable_observatory_scene",
                   return_value=scene):
            with self.assertRaises(ValueError):
                revalidate_observatory_background(self.source)

    def test_saved_evidence_labels_single_gpu_take(self):
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder)/"source.native.npz"
            output = Path(folder)/"observatory"
            save_native_terrain_result(self.source, source_path)
            result, report = prepare_observatory_artifacts(source_path, output)
            self.assertTrue(report["no_second_gpu_generation"])
            self.assertEqual(report["source_native_sha256"],
                             report["observatory_native_sha256"])
            self.assertTrue((output/"observatory.scene.json").is_file())
            self.assertFalse(any(obj["id"].startswith("temple-")
                                 for obj in result.scene["objects"]))
            self.assertEqual(load_native_terrain_result(
                output/"observatory.native.npz").action_spans,
                self.source.action_spans)
            self.assertTrue(json.loads((output/"observatory.report.json").read_text())
                            ["no_second_gpu_generation"])


if __name__ == "__main__":
    unittest.main()
