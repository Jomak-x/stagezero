"""Analytical diagnostics tests: measurements, not learned-motion quality gates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from experiments.grounded_quality import analyze_native, analyze_skinned_feet, analyze_fitted_pair, load_archive, seam_diagnostics, segment_distance


def skeleton(frames=10,actors=1):
    pose=np.random.default_rng(41).normal(size=(27,3))*.25
    pose[:,1]+=1.
    pose[0]=[0,1,0]
    pose[22]=[-.1,.02,0];pose[26]=[.1,.02,0]
    p=np.broadcast_to(pose,(frames,actors,27,3)).copy()
    for actor in range(actors): p[:,actor,:,0]+=actor*2.
    return p


class GroundedQualityTests(unittest.TestCase):
    def test_sliding_is_not_used_to_exclude_contact(self):
        p=skeleton()
        p[:,:,:,0]+=np.arange(10)[:,None,None]*.1
        result=analyze_native(p,schema='core27',fps=20)
        foot=result['actors'][0]['feet']['left']
        self.assertEqual(foot['inferred_contact_intervals'],9)
        self.assertAlmostEqual(foot['horizontal_speed_during_inferred_contact_m_s']['mean'],2.)
        self.assertAlmostEqual(foot['episodes'][0]['path_drift_m'],.9)
        self.assertFalse(result['assumptions']['horizontal_speed_used_to_infer_contact'])
        self.assertNotIn('passed',result)

    def test_airborne_foot_has_no_contact_and_json_null_statistics(self):
        p=skeleton();p[:,:,[22,26],1]=1.
        result=analyze_native(p,schema='core27',fps=20)
        foot=result['actors'][0]['feet']['right']
        self.assertEqual(foot['inferred_contact_intervals'],0)
        self.assertIsNone(foot['horizontal_speed_during_inferred_contact_m_s']['mean'])
        json.dumps(result,allow_nan=False)

    def test_rigid_root_transport_and_turn_do_not_count_as_articulation(self):
        p=skeleton()
        for frame in range(len(p)):
            angle=frame*.17;c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            p[frame]=p[frame]@rotation.T+[frame*.2,0.,0.]
        report=analyze_native(p,schema='core27',fps=20)
        angles=report['actors'][0]['articulation']['internal_joint_angles']
        self.assertLess(max(a['range_degrees'] for a in angles.values()),1e-5)
        self.assertGreater(report['actors'][0]['root_path_length_m'],1.)

    def test_actual_elbow_change_registers_articulation_without_root_motion(self):
        p=skeleton()
        p[:,:,8]=[0.,1.,0.];p[:,:,9]=[0.,1.,1.]
        for i,angle in enumerate(np.linspace(0,np.pi/2,len(p))):
            p[i,:,10]=[np.sin(angle),1.,1.+np.cos(angle)]
        report=analyze_native(p,schema='core27',fps=20)
        self.assertAlmostEqual(report['actors'][0]['articulation']['internal_joint_angles']['right_elbow']['range_degrees'],90.)
        self.assertEqual(report['actors'][0]['root_path_length_m'],0.)

    def test_exact_segment_distances_cross_parallel_and_degenerate(self):
        a=np.array([[0,0,0],[0,0,0],[1,2,3],[0,0,0]])
        b=np.array([[2,0,0],[2,0,0],[1,2,3],[1,0,0]])
        c=np.array([[1,-1,0],[0,1,0],[1,2,5],[2,2,0]])
        d=np.array([[1,1,0],[2,1,0],[1,2,5],[2,3,0]])
        np.testing.assert_allclose(segment_distance(a,b,c,d),[0,1,2,np.sqrt(5)])

    def test_pair_contact_dwell_keeps_same_hand_pair(self):
        p=skeleton(10,2)
        p[:,0,10]=[0.,1.,0.];p[:,1,16]=[2.,1.,0.]
        p[3:7,1,16]=[.04,1.,0.]
        pair=analyze_native(p,schema='core27',fps=20)['pairs'][0]
        contact=pair['wrist_proximity']['right_to_left']['thresholds']['0.1']
        self.assertEqual(contact['frames'],4)
        self.assertAlmostEqual(contact['longest_same_pair_seconds'],.2)

    def test_capsule_overlap_is_reported_separately_from_wrist_contact(self):
        p=skeleton(10,2);p[:,1]=p[:,0]
        overlap=analyze_native(p,schema='core27',fps=20)['pairs'][0]['capsule_overlap']
        self.assertEqual(overlap['frames_with_any_overlap'],10)
        self.assertGreater(overlap['max_depth_per_frame_m']['max'],.25)
        p[:,1,:,0]+=10.
        clear=analyze_native(p,schema='core27',fps=20)['pairs'][0]['capsule_overlap']
        self.assertEqual(clear['frames_with_any_overlap'],0)

    def test_shared_planar_seam_alignment_preserves_pair_geometry_and_arrays(self):
        previous=skeleton(4,2);following=previous.copy()
        angle=.8;c,s=np.cos(angle),np.sin(angle);r=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        following=following@r.T+[5.,0.,-4.]
        before=following.copy()
        candidate=seam_diagnostics(previous,following,schema='core27',fps=20)['best_geometric_candidates'][0]
        self.assertLess(candidate['aligned_joint_step_m']['max'],1e-12)
        self.assertGreater(candidate['raw_joint_step_m']['mean'],1.)
        np.testing.assert_array_equal(following,before)
        following[:,1,:,0]+=.8
        candidate=seam_diagnostics(previous,following,schema='core27',fps=20)['best_geometric_candidates'][0]
        self.assertGreater(candidate['aligned_joint_step_m']['mean'],.1)

    def test_archive_reads_raw_intergen_not_smoothed_and_normalizes_core_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pair.npz'
            raw=np.zeros((8,2,22,3))
            np.savez(path,joints=raw,smoothed_joints=raw+99,metadata=json.dumps({'fps':30}))
            p,schema,fps,_=load_archive(path)
            np.testing.assert_array_equal(p,raw);self.assertEqual(schema,'intergen22');self.assertEqual(fps,30)
            path=Path(directory)/'core.npz';core=skeleton(8,2)
            np.savez(path,positions=core.transpose(1,0,2,3),metadata=json.dumps({'fps':20}))
            p,schema,fps,_=load_archive(path)
            np.testing.assert_array_equal(p,core);self.assertEqual(schema,'core27')

    def test_exact_sole_samples_distinguish_sliding_from_hovering(self):
        class Rig:
            vertices=np.array([[-.2,0,0],[-.2,.005,.1],[.2,0,0],[.2,.005,.1]])
            weights=np.zeros((4,17));weights[:2,11]=1;weights[2:,14]=1
            provenance={'kind':'analytical rigid-foot fixture'}
            def retarget(self,p,r):return {'positions':p,'rotations':r}
            def deform_vertices(self,p,r):return self.vertices+p[0]
        p=skeleton(4)[:,0];p[:,0]=np.array([[i*.05,0,0] for i in range(4)])
        r=np.broadcast_to(np.eye(3),(4,27,3,3))
        result=analyze_skinned_feet(Rig(),p,r,fps=20)
        self.assertAlmostEqual(result['feet']['left']['contact_vertex_horizontal_speed_m_s']['mean'],1.)
        p[:,0,1]+=.1
        result=analyze_skinned_feet(Rig(),p,r,fps=20)
        self.assertEqual(result['feet']['left']['frames_all_sampled_sole_above_5cm'],4)
        self.assertIsNone(result['feet']['left']['contact_vertex_horizontal_speed_m_s']['mean'])
        fitted_p=np.zeros((4,17,3));fitted_p[:,:,1]=.2
        fitted_r=np.broadcast_to(np.eye(3),(4,17,3,3))
        result=analyze_skinned_feet(Rig(),p,r,fps=20,fitted={'positions':fitted_p,'rotations':fitted_r})
        self.assertAlmostEqual(result['feet']['left']['lowest_sole_vertex_each_frame_m']['median'],.2)
        self.assertEqual(result['fitted_pose_input'],'provided fitted transforms')

    def test_fitted_hand_diagnostics_expose_a_grasp_lost_in_retargeting(self):
        class Rig:
            vertices=np.array([[0.,0.,0.],[0.,0.,.01],[0.,0.,.02],[0.,0.,.03]])
            weights=np.zeros((4,17));weights[:2,5]=1;weights[2:,8]=1
            provenance={'kind':'analytical hand-offset fixture'}
            def __init__(self,offset):self.offset=np.array([offset,0.,0.])
            def retarget(self,p,r):
                fitted=np.zeros((17,3));fitted[5]=p[16]+self.offset;fitted[8]=p[10]+self.offset
                return {'positions':fitted,'rotations':r}
            def deform_vertices(self,p,r):return self.vertices+np.array([p[5],p[5],p[8],p[8]])
        p=skeleton(4,2);p[:,0,16]=[0.,1.,0.];p[:,1,16]=[.02,1.,0.]
        r=np.broadcast_to(np.eye(3),(4,2,27,3,3))
        report=analyze_fitted_pair([Rig(0.),Rig(.5)],p,r,fps=20)
        contact=report['hand_pairs']['left_to_left']
        self.assertAlmostEqual(contact['raw_wrist_gap_m']['min'],.02)
        self.assertAlmostEqual(contact['fitted_wrist_gap_m']['min'],.52)
        self.assertAlmostEqual(contact['sampled_hand_vertex_gap_m']['min'],.52)
        self.assertEqual(contact['frames_sampled_hand_gap_under_5cm'],0)

    def test_invalid_and_single_frame_inputs(self):
        with self.assertRaises(ValueError): analyze_native(skeleton(),schema='core27',fps=0)
        with self.assertRaises(ValueError): analyze_native(skeleton()+np.nan,schema='core27',fps=20)
        json.dumps(analyze_native(skeleton(1),schema='core27',fps=20),allow_nan=False)

if __name__=='__main__':unittest.main()
