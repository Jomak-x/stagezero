"""Real loopback HTTP transport tests, with synthetic adapter data and no GPU."""
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import io
import json
import threading
import time
from types import SimpleNamespace
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from interaction_runtime import GenerationCancelled
from realtime_backend import JobManager, encode_chunk, make_handler
from realtime_client import MotionServiceError, RealtimeClient, request_body
from realtime_clip import CanonicalClip


def body(request_id='test-job', **extra):
    result = dict(request_id=request_id, stage_kind='approach', frames=80,
                  prompt='Synthetic HTTP test fixture.', actor_ids=['left', 'right'], seed=7)
    result.update(extra)
    return result


def arrays(index=0):
    # Deliberately simple transport fixtures, not generated model motion.
    p = np.zeros((2,40,27,3),dtype=np.float32)
    p[0,...,0] = -1.2 + index
    p[1,...,0] = 1.2 + index
    r = np.broadcast_to(np.eye(3,dtype=np.float32),(2,40,27,3,3)).copy()
    f = np.arange(2*40*330,dtype=np.float32).reshape(2,40,330) / 1000 + index
    return {'positions':p,'rotations':r,'native_features':f}


class Adapter:
    def __init__(self, mode='normal'):
        self.mode=mode
        self.entered=threading.Event()
        self.release=threading.Event()
        self.jobs=[]
        self.http_paths=[]

    def block(self, job, cancelled, deadline):
        while not self.release.wait(.002):
            if cancelled(job['request_id']):
                raise GenerationCancelled('Test adapter cancelled')
            if time.monotonic() > deadline:
                raise TimeoutError('Test adapter deadline')

    def generate(self,job,on_chunk,is_cancelled,deadline):
        self.jobs.append(job)
        self.entered.set()
        if self.mode=='blocked':
            self.block(job,is_cancelled,deadline)
        if self.mode=='failure':
            raise RuntimeError('Intentional adapter failure')
        for index in range(job['frames']//40):
            if self.mode=='skip' and index==0:
                continue
            aid = tuple(reversed(job['actor_ids'])) if self.mode=='wrong_actors' else job['actor_ids']
            rid = 'wrong-id' if self.mode=='wrong_id' else job['request_id']
            payload = encode_chunk(arrays(index),request_id=rid,index=index,stage_kind=job['stage_kind'],actor_ids=aid)
            if self.mode=='bad_archive':
                payload=b'not an npz archive'
            if self.mode=='bad_header':
                header=io.BytesIO()
                np.lib.format.write_array_header_1_0(header,{
                    'descr':'<f4','fortran_order':False,'shape':(2,1_000_000_000,27,3)})
                output=io.BytesIO()
                with ZipFile(io.BytesIO(payload)) as original, ZipFile(output,'w',ZIP_DEFLATED) as rewritten:
                    for name in original.namelist():
                        rewritten.writestr(name,header.getvalue() if name=='positions.npy' else original.read(name))
                payload=output.getvalue()
            on_chunk(index,payload)
            if self.mode=='partial':
                self.block(job,is_cancelled,deadline)
        return {'test_fixture':True}


@contextmanager
def service(adapter=None, *, post_delay=0, status_delay=0, slow_body=False, redirect=False):
    adapter=adapter or Adapter()
    manager=JobManager(adapter,job_timeout=3)
    base=make_handler(manager,'test-token')
    class Handler(base):
        def do_GET(self):
            adapter.http_paths.append(self.path)
            if redirect and self.path=='/health':
                self.send_response(302)
                self.send_header('Location','/redirect-destination')
                self.send_header('Content-Length','0')
                self.end_headers()
                return
            return super().do_GET()

        def reply(self,status,data,content_type='application/json'):
            if self.command=='POST' and status==202 and post_delay:
                time.sleep(post_delay)
            if self.command=='GET' and '/jobs/' in self.path and '/chunks/' not in self.path and status_delay:
                time.sleep(status_delay)
            if slow_body and self.command=='GET' and '/jobs/' in self.path:
                payload=json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(payload)))
                self.end_headers()
                try:
                    for offset in range(0,len(payload),4):
                        self.wfile.write(payload[offset:offset+4])
                        self.wfile.flush()
                        time.sleep(.01)
                except (BrokenPipeError,ConnectionResetError):
                    pass
                return
            return super().reply(status,data,content_type)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
    thread.start()
    client=RealtimeClient(f'http://127.0.0.1:{server.server_port}','test-token',timeout=.25,job_timeout=2)
    try:
        yield client,manager,adapter
    finally:
        for rid in tuple(manager.jobs):
            manager.cancel(rid)
        adapter.release.set()
        server.shutdown()
        server.server_close()
        thread.join(1)


def wait_for(manager,request_id,status):
    until=time.monotonic()+1
    while time.monotonic()<until:
        current=manager.snapshot(request_id)
        if current and current['status']==status:
            return current
        time.sleep(.003)
    raise AssertionError(f'Expected {status}, received {manager.snapshot(request_id)}')


class RealtimeHTTPTests(unittest.TestCase):
    def test_auth_health_and_complete_chunks_preserve_actor_order_and_native_features(self):
        with service() as (client,manager,adapter):
            self.assertTrue(client.health()['ready'])
            bad=RealtimeClient(client.url,'wrong-token')
            with self.assertRaisesRegex(MotionServiceError,'401'):
                bad.submit(body('unauthorized'))
            self.assertNotIn('unauthorized',manager.jobs)
            seen=[]
            clips=client.wait(body(),on_chunk=seen.append)
            self.assertEqual(len(clips),2)
            self.assertEqual(seen,clips)
            for index,clip in enumerate(clips):
                self.assertEqual(clip.actor_ids,('left','right'))
                self.assertEqual(clip.source,'ardy_core')
                self.assertEqual(clip.metadata['start_frame'],40*index)
                np.testing.assert_array_equal(clip.positions,arrays(index)['positions'])
                np.testing.assert_array_equal(clip.native_features,arrays(index)['native_features'])
                self.assertFalse(clip.native_features.flags.writeable)

    def test_failure_is_surfaced_and_later_job_recovers(self):
        with service(Adapter('failure')) as (client,manager,adapter):
            with self.assertRaisesRegex(RuntimeError,'Intentional adapter failure'):
                client.wait(body('failed'))
            self.assertEqual(manager.snapshot('failed')['status'],'failed')
            adapter.mode='normal'
            self.assertEqual(len(client.wait(body('recovery',frames=40))),1)

    def test_running_cancel_and_precancelled_request_never_submits(self):
        with service(Adapter('blocked')) as (client,manager,adapter):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                client.wait(body('not-submitted'),cancelled=lambda:True)
            self.assertNotIn('not-submitted',manager.jobs)
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                client.wait(body('cancelled'),cancelled=adapter.entered.is_set)
            wait_for(manager,'cancelled','cancelled')

    def test_callback_failure_cancels_partial_job(self):
        with service(Adapter('partial')) as (client,manager,adapter):
            def broken(_):
                raise ValueError('callback failed')
            with self.assertRaisesRegex(ValueError,'callback failed'):
                client.wait(body('partial'),on_chunk=broken)
            wait_for(manager,'partial','cancelled')
            self.assertEqual(manager.snapshot('partial')['available_chunks'],[0])

    def test_timeout_includes_submit_and_cancels_accepted_unacknowledged_job(self):
        with service(Adapter('blocked'),post_delay=.30) as (client,manager,adapter):
            client.timeout=.5
            client.job_timeout=.05
            started=time.monotonic()
            with self.assertRaises(TimeoutError):
                client.wait(body('post-timeout'))
            self.assertLess(time.monotonic()-started,.25)
            wait_for(manager,'post-timeout','cancelled')

    def test_job_deadline_caps_blocking_status_request(self):
        with service(Adapter('blocked'),status_delay=.30) as (client,manager,adapter):
            client.timeout=.5
            client.job_timeout=.05
            started=time.monotonic()
            with self.assertRaises(TimeoutError):
                client.wait(body('status-timeout'))
            self.assertLess(time.monotonic()-started,.25)
            wait_for(manager,'status-timeout','cancelled')

    def test_deadline_interrupts_a_response_body_that_keeps_trickling(self):
        with service(Adapter('blocked'),slow_body=True) as (client,manager,adapter):
            client.timeout=.5
            client.job_timeout=.06
            started=time.monotonic()
            with self.assertRaises(TimeoutError):
                client.wait(body('slow-body'))
            self.assertLess(time.monotonic()-started,.25)
            wait_for(manager,'slow-body','cancelled')

    def test_redirect_is_rejected_without_forwarding_bearer_token(self):
        with service(redirect=True) as (client,manager,adapter):
            with self.assertRaisesRegex(MotionServiceError,'302'):
                client.health()
            self.assertEqual(adapter.http_paths,['/health'])

    def test_duplicate_rejection_does_not_cancel_existing_job(self):
        with service(Adapter('blocked')) as (client,manager,adapter):
            client.submit(body('duplicate'))
            self.assertTrue(adapter.entered.wait(1))
            with self.assertRaisesRegex(MotionServiceError,'already exists'):
                client.wait(body('duplicate'))
            self.assertFalse(manager.get('duplicate').cancel_event.is_set())
            adapter.release.set()
            wait_for(manager,'duplicate','complete')

    def test_not_ready_chunk_and_bad_chunk_payloads(self):
        with service(Adapter('blocked')) as (client,manager,adapter):
            client.submit(body('not-ready'))
            with self.assertRaisesRegex(RuntimeError,'not ready'):
                client.chunk('not-ready',0)
        for mode,message in [('skip','skipped'),('wrong_id','different motion chunk'),
                             ('wrong_actors','different actors'),('bad_archive','Invalid motion chunk'),
                             ('bad_header','Invalid motion array')]:
            with self.subTest(mode=mode), service(Adapter(mode)) as (client,manager,adapter):
                with self.assertRaisesRegex((ValueError,RuntimeError),message):
                    client.wait(body(mode))

    def test_url_encoded_request_id_roundtrip(self):
        with service() as (client,manager,adapter):
            clips=client.wait(body('scene / café % #',frames=40))
            self.assertEqual(len(clips),1)
            self.assertEqual(client.status('scene / café % #')['status'],'complete')
            self.assertEqual(client.cancel('scene / café % #')['status'],'complete')

    def test_native_history_is_unchanged_through_real_http_boundary(self):
        with service() as (client,manager,adapter):
            first=client.wait(body('first',frames=40))[0]
            request=SimpleNamespace(request_id='next',stage_kind='action',source='ardy_core',
                                    history=first,frames=40,prompt='Continue',actor_ids=('left','right'),
                                    actor_prompts=None,metadata={})
            next_body=request_body(request)
            self.assertEqual(next_body['stage_kind'],'continuation')
            self.assertNotIn('initial_placements',next_body)
            client.wait(next_body)
            np.testing.assert_array_equal(adapter.jobs[-1]['history']['native_features'],first.native_features)
            request.actor_ids=('right','left')
            with self.assertRaisesRegex(ValueError,'actor order'):
                request_body(request)

    def test_initial_placements_preserve_explicit_data_and_default_pair_separation(self):
        request=SimpleNamespace(request_id='new',stage_kind='action',source='ardy_core',
                                history=None,frames=40,prompt='Walk',actor_ids=('left','right'),
                                actor_prompts=None,metadata={})
        result=request_body(request)
        self.assertEqual(result['initial_placements'],{
            'left':{'position_xz':[-1.2,0.]},'right':{'position_xz':[1.2,0.]}})
        explicit={'left':{'position_xz':[3.,2.],'yaw':1.2},'right':{'position_xz':[5.,2.]}}
        request.metadata={'initial_placements':explicit}
        self.assertEqual(request_body(request)['initial_placements'],explicit)
        with service() as (client,manager,adapter):
            client.wait(request_body(request))
            actors=adapter.jobs[0]['core_request']['actors']
            self.assertEqual(actors[0]['initial_position_xz'],[3.,2.])
            self.assertEqual(actors[0]['initial_yaw'],1.2)
            self.assertEqual(actors[1]['initial_position_xz'],[5.,2.])

    def test_invalid_urls_and_nonfinite_timeouts_are_rejected(self):
        for url in ['file:///tmp', 'https://user:secret@example.test', 'http://localhost/?key=x','http://localhost/#x']:
            with self.assertRaises(ValueError):
                RealtimeClient(url,'token')
        for timeout in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                RealtimeClient('http://localhost','token',timeout=timeout)


if __name__=='__main__':
    unittest.main()
