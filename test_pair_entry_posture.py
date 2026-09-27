"""CPU checks for prompt-bounded approach posture and Core request parity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import NativePairClip
from native_pair_transition import shared_place_pair
from paired_meetup import build_meetup, plan_meetup
from test_native_pair_rig import fixture_glb
from test_paired_meetup import Client, SCENE, IDS, STARTS, MEETING


class EntryPostureTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'fixture.glb'
        path.write_bytes(fixture_glb()[0])
        pose = NativeRigAsset(path).rest
        self.joints = np.repeat(np.stack([pose + [-1, 0, 0], pose + [1, 0, 0]])[None], 8, axis=0)

    def pair(self, prompt):
        return NativePairClip(self.joints.copy(), metadata={'model': 'InterGen', 'prompt': prompt})

    def plan(self, prompt, policy='continuous'):
        return plan_meetup(self.pair(prompt), SCENE, actor_ids=IDS, starts=STARTS,
                           meeting=MEETING, entry_policy=policy)

    def test_greeting_words_choose_relaxed_entry_without_echoing_prompt(self):
        for prompt in ('Two people greet each other.', 'They shake hands.',
                       'A friendly handshake.', 'They wave hello.', 'They hug.'):
            with self.subTest(prompt=prompt):
                plan = self.plan(prompt)
                self.assertEqual(plan['entry_posture'], 'relaxed')
                cues = [h['actor_prompts'][IDS[0]] for h in plan['horizons']]
                self.assertTrue(all('arms relaxed at their sides' in cue for cue in cues))
                self.assertTrue(all('hands raised' not in cue for cue in cues))
        marker = 'INJECT_UNRELATED_ACTION'
        plan = self.plan('They greet each other. ' + marker)
        self.assertTrue(all(marker not in cue for horizon in plan['horizons']
                            for cue in horizon['actor_prompts'].values()))

    def test_spar_cues_take_priority_when_source_mentions_both_actions(self):
        for prompt in ('Two people spar.', 'A boxing match.',
                       'They exchange punches then shake hands.'):
            with self.subTest(prompt=prompt):
                plan = self.plan(prompt)
                self.assertEqual(plan['entry_posture'], 'guard')
                self.assertTrue(all('boxing guard' in cue for horizon in plan['horizons']
                                    for cue in horizon['actor_prompts'].values()))

    def test_unknown_continuous_prompt_keeps_legacy_ready_cue(self):
        for prompt in ('Two people dance.', '', None, 'The room is a boxing gymnasium'):
            # A scene background should not override an unrelated action cue.
            with self.subTest(prompt=prompt):
                plan = self.plan(prompt)
                self.assertEqual(plan['entry_posture'], 'ready')
                for horizon in plan['horizons']:
                    for cue in horizon['actor_prompts'].values():
                        self.assertIn('hands raised in a ready stance', cue)

    def test_settled_arrival_uses_same_posture_while_unknown_keeps_old_text(self):
        greeting = self.plan('Two people greet each other.', policy='settled')
        spar = self.plan('Two people spar.', policy='settled')
        unknown = self.plan('Two people dance.', policy='settled')
        for aid in IDS:
            self.assertIn('arms relaxed at their sides', greeting['horizons'][-1]['actor_prompts'][aid])
            self.assertIn('boxing guard', spar['horizons'][-1]['actor_prompts'][aid])
            self.assertEqual(unknown['horizons'][-1]['actor_prompts'][aid],
                             'A person stands in place and turns to face their partner, relaxed and ready.')

    def test_build_sends_planned_cues_and_preserves_native_source(self):
        pair = self.pair('Two people greet each other with a handshake.')
        plan = plan_meetup(pair, SCENE, actor_ids=IDS, starts=STARTS, meeting=MEETING,
                           entry_policy='continuous')
        placement = plan['placement']
        client = Client(shared_place_pair(pair.joints,
                                          translation=[placement['x'], 0, placement['z']]))
        result = build_meetup(pair, client, SCENE, actor_ids=IDS, starts=STARTS,
                              meeting=MEETING, entry_policy='continuous')
        self.assertEqual(result['plan']['entry_posture'], 'relaxed')
        self.assertEqual([request['actor_prompts'] for request in client.requests],
                         [horizon['actor_prompts'] for horizon in result['plan']['horizons']])
        np.testing.assert_array_equal(result['clip'].joints[-pair.frames:], pair.joints)
        np.testing.assert_array_equal(pair.joints, self.joints)


if __name__ == '__main__':
    unittest.main()
