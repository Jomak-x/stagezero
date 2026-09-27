import unittest
from pathlib import Path
import numpy as np
from motion_quality import JOINT_INDEX, ROOT, SHOULDERS
from story_action_execution import action_spec, action_prompt, action_completion_frame, action_finished, action_motion_target, generate_action_chunk, directional_motion_target, action_sample_suitable


def standing(frames):
    p=np.zeros((frames,34,3),dtype=float)
    p[:,:,1]=1.0
    for side,x in [('left',-.12),('right',.12)]:
        p[:,JOINT_INDEX[f'{side}_hip_yaw_skel']]=[x,1.,0.]
        p[:,JOINT_INDEX[f'{side}_knee_skel']]=[x,.5,0.]
        p[:,JOINT_INDEX[f'{side}_ankle_roll_skel']]=[x,0.,0.]
    p[:,SHOULDERS,1]=1.5
    return p


class ActionCompletionTests(unittest.TestCase):
    def test_real_pod_actions_are_not_cut_at_old_planned_lengths(self):
        with np.load(Path(__file__).parent/'tests/fixtures/scene_action_completion.npz', allow_pickle=False) as data:
            for name, prompt, old_length in [('fall','A person falls to the ground.',30),
                                             ('backflip','A person performs a backflip.',38)]:
                poses=data[name]; spec=action_spec(prompt)
                self.assertIsNone(action_completion_frame(spec,poses[:old_length]))
                endpoint=action_completion_frame(spec,poses)
                self.assertIsNotNone(endpoint)
                self.assertGreater(endpoint,old_length)
                self.assertTrue(action_finished(spec,poses[:endpoint]))

    def test_travel_requires_requested_displacement(self):
        spec=action_spec('A person sprints forward for 20 meters.')
        p=standing(101);p[:,:,2]+=np.linspace(0,21,101)[:,None]
        self.assertIsNone(action_completion_frame(spec,p[:25]))
        self.assertEqual(action_completion_frame(spec,p),97)
        self.assertTrue(action_finished(spec,p))

    def test_fall_cannot_finish_while_standing(self):
        spec=action_spec('A person falls to the ground.')
        p=standing(40)
        self.assertIsNone(action_completion_frame(spec,p))
        p[20:,ROOT,1]=.2
        p[20:,SHOULDERS,1]=.2
        p[20:,SHOULDERS,2]=.5
        self.assertEqual(action_completion_frame(spec,p),30)
        self.assertTrue(action_finished(spec,p))
        p[-1]=standing(1)[0]
        self.assertFalse(action_finished(spec,p))

    def test_flip_needs_inversion_and_upright_landing(self):
        spec=action_spec('A person performs a backflip.')
        p=standing(90);base=p.copy()
        for i,angle in enumerate(np.linspace(0,2*np.pi,60),10):
            c,s=np.cos(angle),np.sin(angle)
            rot=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            p[i]=(base[i]-base[i,ROOT])@rot.T+base[i,ROOT]
            p[i,:,1]+=.5*np.sin(np.pi*(i-10)/60)
        self.assertIsNone(action_completion_frame(spec,p[:30]))
        self.assertIsNone(action_completion_frame(spec,p[:45]))
        self.assertIsNotNone(action_completion_frame(spec,p))
        self.assertTrue(action_finished(spec,p))
        self.assertIsNone(action_completion_frame(spec,standing(90)))

    def test_stand_after_squat_requires_terminal_upright(self):
        prompt='A person stands upright.'
        self.assertIn('straightens both legs',action_prompt(prompt,'A person squats down.'))
        spec=action_spec(prompt);p=standing(60);p[:40,ROOT,1]=.2
        self.assertEqual(action_completion_frame(spec,p),50)
        self.assertFalse(action_finished(spec,p[:30]))

    def test_stop_needs_settled_root(self):
        p=standing(60);p[:40,:,2]+=np.arange(40)[:,None]*.05;p[40:,:,2]+=1.95
        spec=action_spec('A person stops quickly.')
        self.assertIsNone(action_completion_frame(spec,p[:30]))
        self.assertIsNotNone(action_completion_frame(spec,p))
        p[-15:,ROOT,1]=.2
        self.assertIsNone(action_completion_frame(spec,p))
        self.assertFalse(action_finished(spec,p))

    def test_stop_aliases_reject_upright_motion_until_root_settles(self):
        moving=standing(60)
        moving[:,:,2]+=np.arange(60)[:,None]*.05
        settled=moving.copy()
        settled[40:,:,2]=moving[39,:,2]
        for prompt in ('A person stands still.', 'A person is standing still.',
                       'A person stops and stands upright.'):
            with self.subTest(prompt=prompt):
                spec=action_spec(prompt)
                self.assertEqual(spec,{'kind':'stop'})
                self.assertEqual(action_prompt(prompt),'A person stops and stands still.')
                self.assertIsNone(action_completion_frame(spec,moving))
                self.assertFalse(action_finished(spec,moving))
                self.assertIsNotNone(action_completion_frame(spec,settled))
                self.assertTrue(action_finished(spec,settled))
        self.assertEqual(action_spec('A person stands upright.'),{'kind':'stand'})
        self.assertIsNone(action_spec('A person does not stand still.'))

    def test_native_target_is_bounded_and_optional(self):
        p=standing(104);p[:,:,2]+=np.linspace(0,5,104)[:,None]
        spec=action_spec('A person runs forward for 20 meters.')
        target=action_motion_target(spec,p)
        self.assertEqual(target['frame'],51)
        np.testing.assert_allclose(target['position_xz'],[0.,7.7])
        self.assertIsNone(action_motion_target(spec,None))
        self.assertIsNone(action_motion_target({'kind':'backflip'},p))
        class Supported:
            supports_motion_target=True
            def generate(self,*args,**kwargs):return kwargs
        class Legacy:
            def generate(self,*args):return 'legacy'
        self.assertEqual(generate_action_chunk(Supported(),'id','run',None,spec,p),{'motion_target':target})
        self.assertEqual(generate_action_chunk(Legacy(),'id','run',None,spec,p),'legacy')

    def test_sidesteps_use_body_relative_native_target(self):
        p=standing(52)
        left=directional_motion_target('A person sidesteps left.',None,p)
        right=directional_motion_target('A person sidesteps right.',None,p)
        self.assertLess(left['position_xz'][0],0)
        self.assertGreater(right['position_xz'][0],0)
        self.assertIsNone(directional_motion_target('A person sidesteps left and right.',None,p))
        self.assertIsNone(directional_motion_target('A person does not sidestep left.',None,p))

    def test_finite_stunts_use_bounded_native_profiles(self):
        class Supported:
            supports_generation_options=True
            def generate(self,*args,**kwargs):return kwargs
        backend=Supported()
        options=generate_action_chunk(backend,'id','flip',None,{'kind':'backflip'})
        self.assertEqual(options,{'generation_options':{'profile':'responsive','candidates':3}})
        p=standing(52)
        self.assertEqual(generate_action_chunk(backend,'id','fall',None,{'kind':'fall'},prior_positions=p),{})
        p[:,18:26,0]+=np.sin(np.arange(52)*.3)[:,None]*.2
        p[:,26:34,0]-=np.sin(np.arange(52)*.3)[:,None]*.2
        options=generate_action_chunk(backend,'id','fall',None,{'kind':'fall'},prior_positions=p)
        self.assertEqual(options,{'generation_options':{'profile':'expressive','candidates':3}})
        p[-25:]=standing(25)
        self.assertEqual(generate_action_chunk(backend,'id','fall',None,{'kind':'fall'},prior_positions=p),{})

    def test_backward_walk_target_follows_initial_facing(self):
        p=standing(52)
        for side in ('left','right'):
            p[:,JOINT_INDEX[f'{side}_toe_base']]=p[:,JOINT_INDEX[f'{side}_ankle_roll_skel']]+[0,0,.2]
        target=directional_motion_target('A person walks backward.',None,p)
        np.testing.assert_allclose(target['position_xz'],[0.,-1.2])
        self.assertIsNone(directional_motion_target('A person walks forward.',None,p))
        self.assertIsNone(directional_motion_target('A person walks backward.',None,None))

    def test_dance_freeze_rejected_without_affecting_intentional_holds(self):
        p=standing(104)
        self.assertFalse(action_sample_suitable('A person dances energetically.',p))
        self.assertTrue(action_sample_suitable('A person holds a dance pose.',p))
        p[:,18:26,0]+=np.sin(np.arange(104)*.3)[:,None]*.2
        p[:,26:34,0]-=np.sin(np.arange(104)*.3)[:,None]*.2
        self.assertTrue(action_sample_suitable('A person dances energetically.',p))

    def test_first_action_direction_uses_initial_accepted_pose_without_prior_beat(self):
        p=standing(104)
        for side in ('left','right'):
            p[:,JOINT_INDEX[f'{side}_toe_base']]=p[:,JOINT_INDEX[f'{side}_ankle_roll_skel']]+[0,0,.2]
        p[:,:,2]+=np.linspace(0,-5,104)[:,None]
        # Changing the final foot orientation must not reverse the action's
        # original body frame when selecting the next waypoint.
        for side in ('left','right'):
            p[-1,JOINT_INDEX[f'{side}_toe_base']]=p[-1,JOINT_INDEX[f'{side}_ankle_roll_skel']]+[0,0,-.2]
        target=directional_motion_target('A person walks backward.',p,None)
        np.testing.assert_allclose(target['position_xz'],[0.,-6.2])
        left=directional_motion_target('A person sidesteps left.',p,None)
        np.testing.assert_allclose(left['position_xz'],[-1.2,-5.])
        class Supported:
            supports_motion_target=True
            def generate(self,*args,**kwargs):return kwargs
        sent=generate_action_chunk(Supported(),'id','A person walks backward.',
                                   np.zeros((52,414)),positions=p)
        self.assertEqual(sent,{'motion_target':target})

    def test_unsupported_or_negated_motion_has_no_claim(self):
        self.assertIsNone(action_spec('A person dances.'))
        self.assertIsNone(action_spec('A person does not fall.'))
        self.assertIsNone(action_spec('A person waves.'))

if __name__=='__main__':unittest.main()
