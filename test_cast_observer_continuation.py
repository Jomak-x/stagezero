"""Independent observer controller composition and failure provenance contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from cast_observer_continuation import continue_released_observers
from cast_observer_motion import refine_observers
from cast_observer_turn import ObserverTurnRejected
from test_cast_observer_motion import pose


class ObserverContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.ids = ['a', 'b', 'c']
        self.p = np.repeat(np.stack([pose(), pose()+[3, 0, -3], pose()+[5, 0, -3]])[None], 230, axis=0)
        self.segments = [{'source':'intergen','start_frame':0,'end_frame_exclusive':50},
                         {'source':'ardy_core','start_frame':50,'end_frame_exclusive':110},
                         {'source':'intergen','start_frame':110,'end_frame_exclusive':230}]
        self.activities = [dict(s,active_actor_ids=['a','b'] if n==0 else ['b','c'],
                                contact_actor_ids=['a','b'] if n==0 else ([] if n==1 else ['b','c']))
                           for n,s in enumerate(self.segments)]
        self.track = np.repeat(pose()[None],180,axis=0)
        self.track[:90,:,0] += np.linspace(0,.04,90)[:,None]
        self.track[90:,:,0] += .04
        self.report = {'spans':[{'start_frame':0,'end_frame_exclusive':21,'source':'authored_transition'},
                                {'start_frame':21,'end_frame_exclusive':90,'source':'ardy_core'},
                                {'start_frame':90,'end_frame_exclusive':180,'source':'stationary_hold'}]}
    def run_candidate(self, **kwargs):
        return continue_released_observers(self.p,self.ids,self.segments,self.activities,
            client=object(),scene={},folder=self.tmp.name,seed=48,
            scene_checker=lambda *a,**k:{'checked':True}, **kwargs)
    def test_model_turn_preserves_pair_and_only_refines_stationary_tail(self):
        original=self.p.copy()
        with patch('cast_observer_turn.generate_observer_turn',return_value=(self.track,self.report)) as generate:
            out,activities,reports=self.run_candidate()
        generate.assert_called_once()
        np.testing.assert_array_equal(out[:50],original[:50])
        np.testing.assert_array_equal(out[:,1:],original[:,1:])
        np.testing.assert_array_equal(self.p,original)
        refined,_=refine_observers(out,self.ids,activities)
        np.testing.assert_array_equal(refined[50:140,0],out[50:140,0])
        self.assertGreater(np.abs(refined[145:,0]-out[145:,0]).max(),.01)
        self.assertEqual(reports[0]['status'],'accepted_by_mechanical_and_geometry_gates')
        self.assertTrue(Path(reports[0]['candidate_archive']).is_file())
    def test_collision_rejects_and_preserves_candidate_without_changing_performance(self):
        collision=self.track.copy(); collision[40:] = self.p[90,1]
        with patch('cast_observer_turn.generate_observer_turn',return_value=(collision,self.report)):
            out,activities,reports=self.run_candidate()
        np.testing.assert_array_equal(out,self.p)
        self.assertEqual(activities,self.activities)
        self.assertIn('body-proxy overlap',reports[0]['reason'])
        with np.load(reports[0]['candidate_archive']) as archive:
            np.testing.assert_array_equal(archive['joints'][90:,0],self.p[90:,1])
    def test_rejected_model_candidate_and_sources_remain_archived(self):
        sources=[]
        def reject(*a,**kw):
            sources.append('raw-core.npz')
            raise ObserverTurnRejected('not settled',candidate=self.track,report={'reason':'foot motion'})
        with patch('cast_observer_turn.generate_observer_turn',side_effect=reject):
            out,_,reports=self.run_candidate(source_paths=lambda:sources)
        np.testing.assert_array_equal(out,self.p)
        self.assertEqual(reports[0]['source_archives'],['raw-core.npz'])
        self.assertEqual(reports[0]['motion']['reason'],'foot motion')
        saved=json.loads(next(Path(self.tmp.name).glob('*-report.json')).read_text())
        self.assertEqual(saved['status'],'rejected_preserved_existing_observer')
    def test_cancellation_propagates_without_publishing_a_candidate(self):
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            self.run_candidate(cancelled=lambda:True)
    def test_operational_failure_during_worker_wait_propagates_with_record(self):
        from types import SimpleNamespace
        for error in (OSError('worker disconnected'), RuntimeError('generation cancelled')):
            with self.subTest(error=type(error).__name__):
                def fail(*args, **kwargs):
                    raise error
                with self.assertRaises(type(error)):
                    continue_released_observers(self.p,self.ids,self.segments,self.activities,
                        client=SimpleNamespace(wait=fail),scene={},folder=self.tmp.name,seed=48,
                        scene_checker=lambda *a,**k:{'checked':True})
                record=json.loads(next(Path(self.tmp.name).glob('*-report.json')).read_text())
                self.assertEqual(record['status'],'failed_generation')
                self.assertEqual(record['motion']['rejection_reasons'],[str(error)])

    def test_controller_ranges_cannot_edit_outside_activity(self):
        self.activities[1]['observer_control_spans']=[{'actor_id':'a','start_frame':0,'end_frame_exclusive':65}]
        with self.assertRaisesRegex(ValueError,'outside'):
            refine_observers(self.p,self.ids,self.activities)

if __name__=='__main__': unittest.main()
