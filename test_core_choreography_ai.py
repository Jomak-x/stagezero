"""Offline checks for bounded AI choreography proposals and fail-closed saving."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import requests

from core_choreography_ai import ChoreographyPlanner, main, save_candidate
from object_generation import GatewayGenerator, MAX_RESPONSE_BYTES


def candidate():
    return {'version': 1, 'name': 'Call and response', 'seed': 42, 'beats': [
        {'name': 'Invite', 'seconds': 4, 'actor_prompts': {
            'actor_1': 'Stand in place and raise one hand in greeting.',
            'actor_2': 'Stand in place and nod gently toward the partner.'}},
        {'name': 'Reply', 'seconds': 4, 'actor_prompts': {
            'actor_1': 'Stand in place and lower the hand smoothly.',
            'actor_2': 'Stand in place and raise one hand in reply.'}}]}


class Response:
    def __init__(self, doc=None, *, raw=None, status=200):
        self.body = raw if raw is not None else json.dumps({
            'choices': [{'message': {'content': json.dumps(doc)}}]}).encode()
        self.status_code = status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, _size):
        yield self.body


class ChoreographyAI(unittest.TestCase):
    def planner(self, response=None, error=None):
        transport = mock.Mock()
        transport.post.return_value = response or Response(candidate())
        transport.post.side_effect = error
        return ChoreographyPlanner(GatewayGenerator(
            'https://gateway.example/v1', 'test-model', 'private-token', transport)), transport

    def test_shared_candidate_uses_bounded_existing_transport(self):
        response = Response(candidate())
        planner, transport = self.planner(response)
        result = planner.generate('  Friendly duet  ')
        self.assertEqual(result, candidate())
        self.assertTrue(response.closed)
        transport.post.assert_called_once()
        args, kw = transport.post.call_args
        self.assertEqual(args, ('https://gateway.example/v1/chat/completions',))
        self.assertEqual(kw['timeout'], (10, 90))
        self.assertFalse(kw['allow_redirects'])
        self.assertTrue(kw['stream'])
        self.assertEqual(kw['json']['max_tokens'], 2400)
        self.assertEqual(kw['json']['response_format'], {'type': 'json_object'})
        self.assertEqual(json.loads(kw['json']['messages'][1]['content']), {
            'intent': 'Friendly duet', 'actor_ids': ['actor_1', 'actor_2'], 'seed': 42})
        system = kw['json']['messages'][0]['content']
        self.assertIn('no body contact', system)
        self.assertIn('shadowboxing punches and kicks', system)
        self.assertIn('controlled athletic dodges', system)
        self.assertIn('two independent humanoid motion models', system)
        self.assertNotIn('private-token', json.dumps(result))

    def test_bad_actor_coverage_and_schema_are_not_silently_repaired(self):
        cases = []
        doc = candidate(); del doc['beats'][0]['actor_prompts']['actor_2']; cases.append(doc)
        doc = candidate(); doc['beats'][0]['actor_prompts']['outsider'] = 'wave'; cases.append(doc)
        doc = candidate(); doc['scene'] = {'objects': []}; cases.append(doc)
        doc = candidate(); doc['beats'][0]['script'] = 'run()'; cases.append(doc)
        doc = candidate(); doc['beats'][0]['root_offsets'] = {'actor_1': [0, 0], 'actor_2': [0, 0]}; cases.append(doc)
        doc = candidate(); doc['beats'][0]['seconds'] = 3; cases.append(doc)
        doc = candidate(); doc['beats'][0]['actor_prompts']['actor_1'] = 'x' * 501; cases.append(doc)
        doc = candidate(); doc['beats'][1]['name'] = 'Invite'; cases.append(doc)
        doc = candidate(); doc['beats'] = doc['beats'][:1]; cases.append(doc)
        doc = candidate(); doc['beats'] *= 3; cases.append(doc)
        doc = candidate(); doc['seed'] = 123; cases.append(doc)
        doc = candidate(); doc['seed'] = True; cases.append(doc)
        cases.extend([None, [], {}, {'plan': candidate()}])
        for doc in cases:
            with self.subTest(doc=str(doc)[:90]):
                planner, transport = self.planner(Response(doc))
                with self.assertRaises(ValueError):
                    planner.generate('Friendly duet')
                transport.post.assert_called_once()

    def test_bad_intent_actor_ids_and_seed_never_call_gateway(self):
        planner, transport = self.planner()
        for intent in ('', ' ', 'x' * 1601, None):
            with self.subTest(intent=str(intent)[:20]), self.assertRaises(ValueError):
                planner.generate(intent)
        for ids in ([], ['actor_1'], ['actor_1', 'actor_1'], ['actor_1', 'bad id'], ['actor_1', 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                planner.generate('wave', ids)
        for seed in (-1, 2**31, True, 3.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                planner.generate('wave', seed=seed)
        transport.post.assert_not_called()

    def test_json_escaping_limit_fails_before_gateway_call(self):
        gateway = mock.Mock()
        planner = ChoreographyPlanner(gateway)
        for intent in ('"' * 1600, '\x00' * 1600, 'x' + '\n' * 1598 + 'x'):
            with self.subTest(character=repr(intent[:1])), self.assertRaisesRegex(
                    ValueError, 'Choreography request exceeds 2000 characters after JSON escaping'):
                planner.generate(intent)
        gateway.request_json.assert_not_called()

    def test_bounded_quotes_and_control_characters_round_trip(self):
        intent = 'Wave "hello"\nthen reply\twith a nod.'
        planner, transport = self.planner()
        self.assertEqual(planner.generate(intent), candidate())
        prompt = transport.post.call_args.kwargs['json']['messages'][1]['content']
        self.assertEqual(json.loads(prompt)['intent'], intent)

    def test_eight_two_second_beats_are_allowed_for_clear_action_changes(self):
        doc = candidate()
        doc['beats'] = [{'name': f'Beat {i + 1}', 'seconds': 2,
                         'actor_prompts': {'actor_1': 'Throw a solo shadowboxing jab into empty space.',
                                           'actor_2': 'Perform a controlled athletic dodge in place.'}}
                        for i in range(8)]
        planner, transport = self.planner(Response(doc))
        self.assertEqual(planner.generate('Athletic shadowboxing duet'), doc)
        transport.post.assert_called_once()

    def test_custom_ids_must_match_model_response(self):
        doc = candidate()
        for beat in doc['beats']:
            beat['actor_prompts'] = dict(zip(('lead', 'reply'), beat['actor_prompts'].values()))
        planner, _ = self.planner(Response(doc))
        self.assertEqual(planner.generate('duet', ('lead', 'reply')), doc)

    def test_transport_failures_do_not_fallback_or_leak_secret(self):
        cases = [(Response(raw=b'not json'), None),
                 (Response(raw=b'x' * (MAX_RESPONSE_BYTES + 1)), None),
                 (Response(raw=b'private-token', status=503), None),
                 (None, requests.Timeout('private-token'))]
        for response, error in cases:
            with self.subTest(error=type(error).__name__):
                planner, transport = self.planner(response, error)
                with self.assertRaises(ValueError) as failure:
                    planner.generate('wave')
                self.assertNotIn('private-token', str(failure.exception))
                transport.post.assert_called_once()
                if response:
                    self.assertTrue(response.closed)

    def test_factory_uses_configured_asset_model_and_allows_explicit_choice(self):
        gateway = mock.Mock(model='configured')
        with mock.patch('core_choreography_ai.GatewayGenerator.from_env', return_value=gateway) as factory:
            self.assertIs(ChoreographyPlanner.from_env().gateway, gateway)
            factory.assert_called_once_with(stage='assets')
            self.assertEqual(ChoreographyPlanner.from_env(model='chosen-model').gateway.model, 'chosen-model')
            with self.assertRaises(ValueError):
                ChoreographyPlanner.from_env(model=' ')

    def test_saving_replaces_candidate_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nested' / 'candidate.json'
            save_candidate(candidate(), path)
            self.assertEqual(json.loads(path.read_text()), candidate())
            self.assertEqual([p.name for p in path.parent.iterdir()], ['candidate.json'])

    def test_failed_serialization_preserves_existing_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidate.json'
            path.write_text('existing')
            with self.assertRaises(ValueError):
                save_candidate({'bad': float('nan')}, path)
            self.assertEqual(path.read_text(), 'existing')

    def test_cli_invalid_plan_preserves_existing_file(self):
        planner, transport = self.planner(Response({}))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidate.json'
            path.write_text('existing')
            with mock.patch('core_choreography_ai.ChoreographyPlanner.from_env', return_value=planner), \
                    mock.patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as failure:
                main(['wave', '--output', str(path)])
            self.assertEqual(failure.exception.code, 1)
            self.assertEqual(path.read_text(), 'existing')
            transport.post.assert_called_once()

    def test_cli_saves_candidate_but_does_not_claim_tested_motion(self):
        planner, transport = self.planner()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidate.json'
            with mock.patch('core_choreography_ai.ChoreographyPlanner.from_env', return_value=planner), \
                    mock.patch('sys.stdout', new_callable=io.StringIO) as output:
                main(['wave', '--output', str(path)])
            self.assertEqual(json.loads(path.read_text()), candidate())
            self.assertIn('Motion has not been generated or quality-tested', output.getvalue())
            transport.post.assert_called_once()


if __name__ == '__main__':
    unittest.main()
