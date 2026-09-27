"""Fresh fixed-plan stress trials; reserve shared GPU lanes before --generate.

Each attempt is immutable, including failed raw motion. This tests motion and
composition, not fresh external AI scene planning. No provisioning occurs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.review_interaction_v2 import digest, measure, next_attempt, CODE_FILES, _source_manifest
from prompt_scene_plan import validate_plan
REVIEW=ROOT/'review/interaction-demo-final'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true',required=True)
    parser.add_argument('--case',default='all')
    parser.add_argument('--model-lane-reserved',action='store_true')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--token',type=Path,required=True)
    args=parser.parse_args()
    if not args.model_lane_reserved: parser.error('Reserve shared Core/InterGen lane first')
    cases=json.loads((REVIEW/'cases.json').read_text())
    if args.case!='all':
        cases=[c for c in cases if c['case']==args.case]
        if not cases: parser.error('Unknown case')
    for case in cases:
        plan=REVIEW/case['plan']; document=json.loads(plan.read_text())
        scene=ROOT/'review/prompt-scenes/backgrounds'/f'{case["background"]}.json'
        validate_plan(document,json.loads(scene.read_text()),document['prompt'])
        output=next_attempt(REVIEW/'fresh'/case['case'])
        info=dict(case=case,planner='explicit frozen authored test plan; not a fresh AI planner run',
                  plan_sha256=digest(plan),scene_sha256=digest(scene),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  code_file_sha256={n:digest(ROOT/n) for n in CODE_FILES})
        command=[sys.executable,str(ROOT/'experiments/trial_prompt_scene.py'),'--prompt',document['prompt'],
                 '--scene',str(scene),'--plan',str(plan),'--config',str(args.config),'--token',str(args.token),
                 '--seed',str(case['seed']),'--output',str(output)]
        started=time.perf_counter()
        try:
            run=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=240)
            log=run.stdout+run.stderr;code=run.returncode
        except subprocess.TimeoutExpired as exc:
            log='Hard 240s timeout; partial source directory preserved.\n'+str(exc.stdout or '')+str(exc.stderr or '');code=124
        output.mkdir(parents=True,exist_ok=True);(output/'run.log').write_text(log)
        archive=output/'scene.cast.stagezero.npz'
        info.update(runner_wall_seconds=time.perf_counter()-started,exit_code=code,
                    archive_sha256=digest(archive) if archive.exists() else None,
                    source_archives=_source_manifest(archive),raw_failures_preserved=True)
        (output/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
        if archive.exists(): (output/'metrics.json').write_text(json.dumps(measure(archive),indent=2)+'\n')
        else: (output/'failure.json').write_text(json.dumps({'exit_code':code,'last_log_lines':log.splitlines()[-12:]},indent=2)+'\n')
        print(json.dumps({'case':case['case'],'exit_code':code,'archive':archive.exists(),'seconds':info['runner_wall_seconds'],'attempt':str(output.relative_to(ROOT))}),flush=True)

if __name__=='__main__':main()
