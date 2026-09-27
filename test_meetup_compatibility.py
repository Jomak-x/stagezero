"""Keep main's existing acceptance when the optional new clearance policy fails."""
import unittest
from unittest.mock import patch
from prompt_scene_builder import build_compatible_meetup

class MeetupCompatibilityTests(unittest.TestCase):
    def test_only_new_gate_retries_once_with_same_inputs_seed_and_archives(self):
        pair,client,scene=object(),object(),object();paths=[]
        error=ValueError('new approach clearance')
        error.approach_body_clearance_report={'passed':False,'overlap_frames':3,'minimum_clearance_m':-.0583}
        def build(*args,**kwargs):
            self.assertEqual(args,(pair,client,scene));self.assertEqual(kwargs['seed'],64)
            paths.append('core-'+str(len(paths)))
            if len(paths)==1:raise error
            self.assertEqual(kwargs['arrival_margin_m'],0.)
            return {'metadata':{'approach_body_clearance':{'checked':False}},'joints':'unchanged'}
        with patch('prompt_scene_builder.build_meetup',side_effect=build) as call:
            out=build_compatible_meetup(pair,client,scene,arrival_margin_m=.08,seed=64,source_paths=lambda:paths)
        self.assertEqual(call.call_count,2)
        report=out['metadata']['arrival_margin_recovery']
        self.assertEqual(report['enhanced_source_archives'],['core-0'])
        self.assertEqual(report['fallback_source_archives'],['core-1'])
        self.assertFalse(out['metadata']['approach_body_clearance']['checked'])
        self.assertEqual(out['joints'],'unchanged')
    def test_original_rejections_and_worker_failure_do_not_retry(self):
        for error in (ValueError('sampled endpoint velocity exceeds tolerance'),OSError('worker disconnected')):
            with patch('prompt_scene_builder.build_meetup',side_effect=error) as call:
                with self.assertRaises(type(error)):
                    build_compatible_meetup(object(),object(),{},arrival_margin_m=.08)
            self.assertEqual(call.call_count,1)
    def test_retry_failure_preserves_both_attempts_and_propagates(self):
        paths=[];error=ValueError('new approach clearance');error.approach_body_clearance_report={'passed':False}
        failure=ValueError('original geometry gate')
        def build(*a,**kw):
            paths.append(str(len(paths)))
            raise error if len(paths)==1 else failure
        with patch('prompt_scene_builder.build_meetup',side_effect=build) as call:
            with self.assertRaises(ValueError) as caught:
                build_compatible_meetup(object(),object(),{},arrival_margin_m=.08,source_paths=lambda:paths)
        self.assertIs(caught.exception,failure);self.assertEqual(call.call_count,2)
        self.assertEqual(failure.arrival_margin_recovery['fallback_source_archives'],['1'])
    def test_first_success_and_cancellation_remain_bounded(self):
        out={'metadata':{}}
        with patch('prompt_scene_builder.build_meetup',return_value=out) as call:
            self.assertIs(build_compatible_meetup(object(),object(),{},arrival_margin_m=.08),out)
        self.assertEqual(call.call_count,1);self.assertNotIn('arrival_margin_recovery',out['metadata'])
        error=ValueError('new gate');error.approach_body_clearance_report={'passed':False}
        with patch('prompt_scene_builder.build_meetup',side_effect=error) as call:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                build_compatible_meetup(object(),object(),{},arrival_margin_m=.08,cancelled=lambda:True)
        self.assertEqual(call.call_count,1)

if __name__=='__main__':unittest.main()
