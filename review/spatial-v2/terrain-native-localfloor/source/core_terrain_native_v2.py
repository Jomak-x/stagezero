"""Bounded actual Core terrain trials; conditioning only, default FK output.

Three seeds and at most three native-conditioning variants. Save every native
motion tensor and FK pose for visual inspection; no root/foot/rotation edits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--local-floor', action='store_true')
    parser.add_argument('--seeds', default='11,22,33')
    parser.add_argument('--variants', default='height_sparse,height_dense,height_foot')
    args = parser.parse_args()
    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything
    from ardy.constraints import Root2DConstraintSet
    from core_terrain_constraints import RootHeightConstraint, shift_vertical_coordinate_frame
    from scene_interaction_geometry import SceneInteractionGeometry
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads(args.scene.read_text())
    geometry = SceneInteractionGeometry.from_scene(scene)
    stairs = next(s for s in geometry.stair_routes if s['name'] == 'Bridge approach stairs')
    steps = np.asarray(stairs['steps'])
    start = np.array([0., 2.0304636, -5.25]); end = np.array([0., 2.9006623, -8.3])
    route = np.vstack((start, steps, end))
    # Smooth pelvis trajectory through tread centres; foot tests still use the
    # exact rendered horizontal surfaces, never this interpolated ramp.
    def route_at(frame):
        z = np.interp(frame, [0, 140, 159], [start[2], end[2], end[2]])
        y = np.interp(-z, -route[:,2], route[:,1])
        return np.array([0., y, z])
    report = {'status':'loading', 'cases':[], 'route':route.tolist(), 'fps':20, 'local_floor':args.local_floor,
              'warning':'Experimental native outputs; metrics do not prove gait or physics.'}
    def save(): (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    print('LOADING CORE', flush=True)
    model = load_model('ARDY-Core-RP-20FPS-Horizon40', device='cuda', checkpoints_dir=args.checkpoints, text_encoder_mode='local', text_encoder=False if args.embeddings else None)
    names = list(model.skeleton.bone_order_names)
    parents = [p for _,p in model.skeleton.bone_order_names_with_parents]
    report['joint_names']=names; report['parents']=parents
    report['slice_dict']={k:[v.start,v.stop] for k,v in model.motion_rep.slice_dict.items()}
    report['model_steps']=int(model.diffusion.num_base_steps)
    prompt='A person walks up a short flight of shallow stairs, lifting each foot onto the next step, with a relaxed upright posture.'
    print('ENCODING',flush=True)
    with torch.inference_mode():
        embeddings=({k:tuple(v.to('cuda') for v in torch.load(args.embeddings/(k+'_embedding.pt'),weights_only=True)) for k in ('stand','walk')} if args.embeddings else {k:model._encode_text([v]) for k,v in {'stand':'A person stands still in a relaxed neutral pose.', 'walk':prompt}.items()})
    # Free memory once prompt features are captured; only model remains.
    model.text_encoder=None
    import gc
    gc.collect(); torch.cuda.empty_cache()
    for key,value in embeddings.items():
        torch.save(tuple(v.detach().cpu() for v in value),args.output/(key+'_embedding.pt'))
    report['status']='running'; save()
    floor_origin = float(start[1]) if args.local_floor else 0.
    def constraints(frames, xyz, height=True):
        frames=torch.as_tensor(frames,device='cuda',dtype=torch.long)
        xyz=torch.as_tensor(xyz,device='cuda',dtype=torch.float32).clone()
        xyz[:,1] -= floor_origin
        result=[Root2DConstraintSet(model.skeleton, frames, xyz[:,[0,2]])]
        if height: result.append(RootHeightConstraint(frames,xyz[:,1]))
        return result
    def generate(history, cond, key):
        h=0 if history is None else history.shape[1]
        obs,mask=model.motion_rep.create_conditions_from_constraints_batched(cond, torch.tensor([h+40],device='cuda'),True,'cuda')
        assert not mask[:,:h].any()
        emb=embeddings[key]
        with torch.inference_mode():
            motion=model.autoregressive_step(num_frames=h+40,num_denoising_steps=model.diffusion.num_base_steps,
                motion_mask=mask,observed_motion=obs,cfg_weight=(2.,2.),text_feat=emb[0],text_pad_mask=emb[1],
                init_history_sequence=(shift_vertical_coordinate_frame(history,model.motion_rep,-floor_origin) if history is not None and floor_origin else history),init_global_translation=torch.tensor([[0.,0.,start[2]]],device='cuda'),
                init_first_heading_angle=torch.tensor([np.pi],device='cuda',dtype=torch.float32))
        return (shift_vertical_coordinate_frame(motion[:,h:],model.motion_rep,floor_origin) if floor_origin else motion[:,h:]),mask
    seed_everything(8171)
    prefix,_=generate(None,constraints([7,15,23,31,39],np.tile(start+[0,.95,0],(5,1))),'stand')
    np.savez_compressed(args.output/'prefix.npz',motion=prefix[0].cpu().numpy())
    for variant in args.variants.split(','):
        for seed in map(int,args.seeds.split(',')):
            seed_everything(seed); current=prefix[:,-4:].clone(); chunks=[]; audits=[]; began=time.monotonic()
            for window in range(4):
                off=window*40; h=current.shape[1]
                floor_origin=float(route_at(off)[1]) if args.local_floor else 0.
                local=np.arange(40) if variant=='height_dense' else np.array([7,15,23,31,39])
                points=np.asarray([route_at(off+i+1)+[0,.95,0] for i in local])
                cond=constraints(local+h,points)
                if variant=='height_foot':
                    # Position-only native hints intentionally omit ancestry rotations.
                    # Their decoded FK error is measured, not assumed zero.
                    class FootPositions:
                        def update_constraints(self,data,index):
                            frames=np.array([11,27])+h
                            rows=[]; vals=[]
                            for j,frame in enumerate(frames):
                                root=route_at(off+frame-h+1)+[0,.95,0]
                                foot=root.copy(); foot[0]=(-.1 if (off//16+j)%2==0 else .1)
                                foot[2]-=.18
                                support=geometry.support_height(float(foot[0]),float(foot[2]),root[1]-.95,max_step_up=.3,max_drop=.4)
                                foot[1]=(root[1]-.95 if support is None else support)+.06
                                name='LeftToeBase' if foot[0]<0 else 'RightToeBase'
                                rows += [[frame,model.skeleton.root_idx],[frame,names.index(name)]]
                                vals += [root-[0,floor_origin,0],foot-[0,floor_origin,0]]
                            data['global_joints_positions'].append(torch.tensor(np.array(vals),device='cuda',dtype=torch.float32))
                            index['global_joints_positions'].append(torch.tensor(rows,device='cuda'))
                    frames=np.array([11,27]); xyz=np.asarray([route_at(off+i+1)+[0,.95,0] for i in frames])
                    cond += constraints(frames+h,xyz)+[FootPositions()]
                piece,mask=generate(current,cond,'walk'); chunks.append(piece); current=piece[:,-4:]
                audits.append(sorted(torch.nonzero(mask[0].any(0)).flatten().cpu().tolist()))
                print(variant,seed,window,'generated',flush=True)
            full=torch.cat(chunks,1)
            with torch.inference_mode(): out=model.motion_rep.inverse(full,is_normalized=True)
            pos=out['posed_joints'][0].cpu().numpy(); rot=out['global_rot_mats'][0].cpu().numpy()
            root=pos[:,model.skeleton.root_idx]; toes=pos[:,[names.index('LeftToeBase'),names.index('RightToeBase')]]
            support=np.full((160,2),np.nan)
            for f in range(160):
                for j in range(2):
                    s=geometry.support_height(float(toes[f,j,0]),float(toes[f,j,2]),float(route_at(f+1)[1]),max_step_up=.4,max_drop=.4)
                    if s is not None: support[f,j]=s
            clearance=toes[:,:,1]-support
            torso=pos[:,names.index('Head')]-root
            lean=np.degrees(np.arctan2(np.linalg.norm(torso[:,[0,2]],axis=1),torso[:,1]))
            footstep=np.linalg.norm(np.diff(toes[:,:,[0,2]],axis=0),axis=-1)*20
            near=(np.abs(clearance[1:])<.10)&(np.abs(clearance[:-1])<.10)
            planned=np.asarray([route_at(i+1)+[0,.95,0] for i in range(160)])
            metrics={'min_toe_surface_clearance_m':float(np.nanmin(clearance)),
                'toe_penetration_fraction_below_minus_5cm':float(np.nanmean(clearance<-.05)),
                'both_feet_airborne_fraction_above_15cm':float(np.mean(np.nanmin(clearance,axis=1)>.15)),
                'unsupported_toe_fraction':float(np.isnan(support).mean()),
                'root_target_mean_error_m':float(np.linalg.norm(root-planned,axis=1).mean()),
                'root_end':root[-1].tolist(),'torso_lean_median_deg':float(np.median(lean)),
                'heuristic_stance_slip_p95_m_s':float(np.percentile(footstep[near],95)) if near.any() else None,
                'root_max_frame_step_m':float(np.linalg.norm(np.diff(root,axis=0),axis=1).max())}
            name=f'{variant}__seed{seed}'
            np.savez_compressed(args.output/(name+'.npz'),motion=full[0].cpu().numpy(),positions=pos,rotations=rot,
                support=support,clearance=clearance,planned_root=planned,contacts=out['foot_contacts'][0].cpu().numpy())
            case={'name':name,'variant':variant,'seed':seed,'seconds':time.monotonic()-began,'metrics':metrics,'mask_feature_indices':audits}
            report['cases'].append(case);save();print(json.dumps(case),flush=True)
    report['status']='completed';save()

if __name__=='__main__': main()
