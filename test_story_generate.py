"""Offline CLI contract checks for full-scene plans."""

import json
from pathlib import Path
import tempfile
import unittest

from story_generate import build_parser, run
from test_story_jobs import plan


class Planner:
    def plan(self, prompt, context=None, seconds=None):
        result = plan()
        result['prompt'] = prompt
        result['beats'][0]['seconds'] = 2.4
        result['beats'][1]['seconds'] = 5.92
        return result


class StoryGenerateTests(unittest.TestCase):
    def test_plan_only_defaults_to_auto_natural_duration_without_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.json'
            args = build_parser().parse_args(['  Walk, then wave.  ', '--output', str(output),
                                              '--plan-only'])
            self.assertIsNone(args.seconds)
            run(args, planner=Planner())
            saved = json.loads(output.read_text())
            self.assertEqual(sum(round(beat['seconds'] * 25) for beat in saved['beats']), 208)
            self.assertEqual([beat['seconds'] for beat in saved['beats']], [2.4, 5.92])
            self.assertEqual(saved['prompt'], 'Walk, then wave.')

    def test_feasible_fixed_duration_writes_exact_plan(self):
        class FixedPlanner(Planner):
            def plan(self, prompt, context=None, seconds=None):
                result = super().plan(prompt, context, seconds)
                result['beats'][0]['seconds'] = 16
                result['beats'][1]['seconds'] = 4
                return result

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.json'
            args = build_parser().parse_args(['Walk, then wave.', '--output', str(output),
                                              '--seconds', '20', '--plan-only'])
            run(args, planner=FixedPlanner())
            saved = json.loads(output.read_text())
            self.assertEqual([beat['seconds'] for beat in saved['beats']], [16, 4])

    def test_fixed_duration_requires_feasible_plan_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.json'
            args = build_parser().parse_args(['Walk, then wave.', '--output', str(output),
                                              '--seconds', '60', '--plan-only'])
            with self.assertRaisesRegex(ValueError, 'use Auto'):
                run(args, planner=Planner())
            self.assertFalse(output.exists())

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
