"""Offline end-to-end orchestration checks for the self-hosted character path."""
import unittest
from unittest.mock import Mock
from io import BytesIO

from PIL import Image

from character_pipeline import SelfHostedCharacterGenerator
from test_character_generation import FakeTransport, FakeResponse, response, glb


class Transport(FakeTransport):
    def delete(self, url, **kwargs):
        return self._call('delete', url, **kwargs)


class PipelineTests(unittest.TestCase):
    @staticmethod
    def png():
        buffer = BytesIO()
        Image.new('RGB', (32, 48), 'gray').save(buffer, format='PNG')
        return buffer.getvalue()

    def build(self, replies, **kwargs):
        reference = Mock()
        reference.generate.return_value = b'PNG fixture'
        transport = Transport(replies)
        generator = SelfHostedCharacterGenerator(reference, 'private-token', transport=transport,
                                                 sleep=lambda _: None, **kwargs)
        return generator, reference, transport

    def test_reference_to_gpu_to_glb(self):
        identifier = 'a' * 32
        generator, reference, transport = self.build([
            response({'ready': True, 'busy': False}), response({'id': identifier}),
            response({'status': 'generating'}), response({'status': 'succeeded'}), FakeResponse(glb())])
        self.assertEqual(generator.generate('A realistic explorer'), glb())
        self.assertEqual(generator.reference_image, b'PNG fixture')
        reference.generate.assert_called_once()
        self.assertEqual(transport.calls[1][2]['data'], b'PNG fixture')
        self.assertTrue(transport.calls[-1][1].endswith('/jobs/' + identifier + '/result'))
        self.assertTrue(all(not call[2]['allow_redirects'] for call in transport.calls))

    def test_worker_not_ready_or_busy_does_not_spend_reference_request(self):
        for health in ({'ready': False}, {'ready': True, 'busy': True}):
            generator, reference, _ = self.build([response(health)])
            with self.assertRaises(ValueError):
                generator.generate('Explorer')
            reference.generate.assert_not_called()

    def test_cancel_existing_gpu_job(self):
        generator, _, transport = self.build([
            response({'ready': True}), response({'id': 'b' * 32}), response({'status': 'cancelled'})])
        with self.assertRaisesRegex(ValueError, 'stopped'):
            generator.generate('Explorer', cancelled=lambda: len(transport.calls) >= 2)
        self.assertEqual(transport.calls[-1][0], 'delete')

    def test_failure_does_not_leak_worker_details(self):
        generator, _, transport = self.build([
            response({'ready': True}), response({'id': 'b' * 32}),
            response({'status': 'failed', 'error': 'secret-token /private/path'}),
            response({'status': 'cancelled'})])
        with self.assertRaisesRegex(ValueError, 'could not finish') as raised:
            generator.generate('Explorer')
        self.assertNotIn('secret', str(raised.exception))
        self.assertEqual(transport.calls[-1][0], 'delete')

    def test_malformed_job_id_cannot_change_route(self):
        generator, _, transport = self.build([response({'ready': True}), response({'id': '../../secret'})])
        with self.assertRaisesRegex(ValueError, 'identifier'):
            generator.generate('Explorer')
        self.assertEqual(len(transport.calls), 2)

    def test_capacity_failure_is_actionable_without_exposing_worker_logs(self):
        generator, _, _ = self.build([
            response({'ready': True}), response({'id': 'b' * 32}),
            response({'status': 'failed', 'error': 'GPU memory busy'}), response({})])
        with self.assertRaisesRegex(ValueError, 'shared GPU is still busy'):
            generator.generate('Explorer')

    def test_failed_gpu_job_can_retry_with_exact_completed_reference_once(self):
        png = self.png()
        first, reference, _ = self.build([
            response({'ready': True}), response({'id': 'a' * 32}),
            response({'status': 'failed', 'error': 'GPU memory busy'}), response({})])
        reference.generate.return_value = png
        with self.assertRaisesRegex(ValueError, 'shared GPU is still busy'):
            first.generate('Explorer')
        reference.generate.assert_called_once()

        retry, retry_reference, transport = self.build([
            response({'ready': True}), response({'id': 'b' * 32}),
            response({'status': 'succeeded'}), FakeResponse(glb()),
            response({'ready': True}), response({'id': 'c' * 32}),
            response({'status': 'succeeded'}), FakeResponse(glb())])
        retry_reference.generate.return_value = png
        seen = []
        retry.on_reference = seen.append
        retry.use_reference('Explorer', png)
        messages = []
        self.assertEqual(retry.generate('Explorer', progress=messages.append), glb())
        retry_reference.generate.assert_not_called()
        self.assertEqual(transport.calls[1][2]['data'], png)
        self.assertEqual(seen, [png])
        self.assertIn('1 / 3 · Reusing your completed design…', messages)
        self.assertEqual(retry.generate('Explorer'), glb())
        retry_reference.generate.assert_called_once()

    def test_seed_rejects_invalid_png_and_different_prompt_uses_neon(self):
        png = self.png()
        generator, reference, transport = self.build([
            response({'ready': True}), response({'id': 'd' * 32}),
            response({'status': 'succeeded'}), FakeResponse(glb())])
        reference.generate.return_value = b'new reference'
        with self.assertRaises(ValueError):
            generator.use_reference('Explorer', b'invalid PNG')
        generator.use_reference('Explorer', png)
        generator.generate('Pilot')
        reference.generate.assert_called_once()
        self.assertEqual(transport.calls[1][2]['data'], b'new reference')

    def test_cancellation_during_result_stream_cleans_up_job(self):
        identifier = 'c' * 32
        cancelled = [False]
        class StreamingResult(FakeResponse):
            def iter_content(self, _size):
                yield glb()[:12]
                cancelled[0] = True
                yield glb()[12:]
        generator, _, transport = self.build([
            response({'ready': True}), response({'id': identifier}),
            response({'status': 'succeeded'}), StreamingResult(), response({})])
        with self.assertRaisesRegex(ValueError, 'stopped'):
            generator.generate('Explorer', cancelled=lambda: cancelled[0])
        self.assertEqual(transport.calls[-1][0], 'delete')
        self.assertTrue(transport.calls[-1][1].endswith('/jobs/' + identifier))

    def test_result_stream_obeys_deadline_and_bounds_request_timeout(self):
        identifier = 'd' * 32
        now = [0.0]
        class SlowResult(FakeResponse):
            def iter_content(self, _size):
                yield glb()[:12]
                now[0] = 1.1
                yield glb()[12:]
        generator, _, transport = self.build([
            response({'ready': True}), response({'id': identifier}),
            response({'status': 'succeeded'}), SlowResult(), response({})],
            clock=lambda: now[0], deadline=1.0)
        with self.assertRaisesRegex(ValueError, 'timed out'):
            generator.generate('Explorer')
        result_call = transport.calls[-2]
        self.assertEqual(result_call[0], 'get')
        self.assertLessEqual(result_call[2]['timeout'][1], 1.0)
        self.assertEqual(transport.calls[-1][0], 'delete')

    def test_job_creation_reads_identifier_before_cancellation_cleanup(self):
        identifier = 'e' * 32
        cancelled = [False]
        encoded = response({'id': identifier}).body
        class StreamingCreation(FakeResponse):
            def iter_content(self, _size):
                yield encoded[:5]
                cancelled[0] = True
                yield encoded[5:]
        generator, _, transport = self.build([
            response({'ready': True}), StreamingCreation(), response({})])
        with self.assertRaisesRegex(ValueError, 'stopped'):
            generator.generate('Explorer', cancelled=lambda: cancelled[0])
        self.assertEqual(transport.calls[-1][0], 'delete')
        self.assertTrue(transport.calls[-1][1].endswith('/jobs/' + identifier))

    def test_bad_prompt_and_remote_backend_rejected(self):
        generator, reference, transport = self.build([])
        for prompt in (None, '', 'x' * 801):
            with self.assertRaises(ValueError):
                generator.generate(prompt)
        reference.generate.assert_not_called()
        self.assertFalse(transport.calls)
        for url in ('http://example.com', 'https://127.0.0.1', 'http://127.0.0.1/path', 'http://user@localhost'):
            with self.assertRaises(ValueError):
                SelfHostedCharacterGenerator(reference, 'token', url)
        for limits in ({'deadline': 0}, {'deadline': float('nan')},
                       {'poll_interval': 0}, {'poll_interval': float('inf')}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                SelfHostedCharacterGenerator(reference, 'token', **limits)


if __name__ == '__main__':
    unittest.main()
