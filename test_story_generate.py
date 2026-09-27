"""Offline CLI contract checks for full-scene plans."""

import json
from pathlib import Path
import tempfile
import unittest

from story_generate import build_parser, run
from test_story_jobs import plan


class Planner:
    def plan(self, prompt, context=None, seconds=None):
        result = plan(seconds)
        result['prompt'] = prompt
        return result


class StoryGenerateTests(unittest.TestCase):
    def test_plan_only_writes_exact_requested_duration_without_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.json'
            args = build_parser().parse_args(['Walk, then wave.', '--output', str(output),
                                              '--seconds', '60', '--plan-only'])
            run(args, planner=Planner())
            saved = json.loads(output.read_text())
            self.assertEqual(sum(round(beat['seconds'] * 25) for beat in saved['beats']), 1500)
            self.assertEqual(saved['prompt'], 'Walk, then wave.')

    def test_rejects_overlong_scene_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.json'
            args = build_parser().parse_args(['Walk, then wave.', '--output', str(output),
                                              '--seconds', '121', '--plan-only'])
            with self.assertRaisesRegex(ValueError, '120'):
                run(args, planner=Planner())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
