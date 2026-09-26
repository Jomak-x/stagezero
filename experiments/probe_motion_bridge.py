#!/usr/bin/env python3
"""CPU-only retarget audit of recorded native clips; never starts a model/GPU.

The optional Core reference check re-encodes saved native poses using official
ArdyMotionRep with NO normalization to measure geometric identity roundtrip.
Production normalization requires a loaded model and is not claimed here.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motion_bridge import core_skeleton, retarget_intergen_pair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--core-reference-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'clips':{},'core_identity':{},'normalization':'not performed; production requires checkpoint statistics'}
    for filename in sorted(args.input_dir.glob('*.npz')):
        with np.load(filename,allow_pickle=False) as archive:
            if 'joints' not in archive:
                continue
            source=archive['joints']
            original=json.loads(str(archive['metadata'])) if 'metadata' in archive else {}
        positions,rotations,metadata=retarget_intergen_pair(source)
        metadata['source_file']=str(filename)
        metadata['source_metadata']=original
        np.savez_compressed(args.output_dir/filename.name,positions=positions,rotations=rotations,metadata=json.dumps(metadata))
        report['clips'][filename.name]={'scale':metadata['common_scale'],'frames':metadata['frames'],**metadata['metrics']}
    if args.core_reference_dir:
        from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
        from ardy.skeleton.transforms import global_rots_to_local_rots
        rep=ArdyMotionRep(core_skeleton(),fps=20)
        for filename in sorted(args.core_reference_dir.glob('*.npz')):
            archive=np.load(filename,allow_pickle=False)
            for key in archive.files:
                if not key.endswith('_positions'):
                    continue
                rkey=key.removesuffix('_positions')+'_rotations'
                if rkey not in archive:
                    continue
                p,r=archive[key],archive[rkey]
                if p.shape[-2:] != (27,3):
                    continue
                local=global_rots_to_local_rots(torch.from_numpy(r)[None],rep.skeleton)
                features=rep(local,torch.from_numpy(p)[None,:,0],to_normalize=False)
                decoded=rep.inverse(features,is_normalized=False)
                report['core_identity'][filename.name+':'+key]={
                    'positions_max_error_m':float(np.max(np.abs(decoded['posed_joints'].numpy()[0]-p))),
                    'rotations_max_error':float(np.max(np.abs(decoded['global_rot_mats'].numpy()[0]-r))),
                    'features':list(features.shape),
                }
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
