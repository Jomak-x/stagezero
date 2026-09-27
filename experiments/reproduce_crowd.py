"""Reproduce fixed-seed crowd trajectories from the recorded AI plan, offline."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from crowd_navigation import simulate_crowd
from crowd_plan import validate_crowd_intent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=ROOT/'review/crowd-crossing/plan.json')
    p.add_argument('--output',type=Path,required=True,help='New directory; existing evidence is never overwritten')
    a=p.parse_args();plan=json.loads(a.plan.read_text());validate_crowd_intent(plan['intent'])
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for count in (16,32,64,100):
        result=simulate_crowd(dict(plan['config'],count=count))
        result.write_json(a.output/f'trajectories-{count}.json')
        results.append({'count':count,**result.metrics})
    (a.output/'solver-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
