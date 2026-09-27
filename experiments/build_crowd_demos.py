"""Build deterministic saved demonstrations; never runs on viewer playback."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crowd_demo_navigation import DemoConfig,generate,validate,apply_city_attendant,apply_crossing_queue_cues

def main():
    p=argparse.ArgumentParser();p.add_argument('--counts',nargs='+',type=int,default=[112]);p.add_argument('--scenes',nargs='+',choices=['crossing','city','station'],default=['crossing','city','station']);p.add_argument('--manifest',type=Path,default=Path('review/crowd-demos/assets/manifest.json'));p.add_argument('--output',type=Path,default=Path('review/crowd-demos'));args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    metrics={}
    for kind in args.scenes:
        for count in args.counts:
            data=generate(DemoConfig(environment=kind,count=count));
            from crowd_demo_cues import apply_motion_cues
            manifest=args.manifest
            if manifest.exists():
                atlas=json.loads(manifest.read_text());data=apply_motion_cues(data,atlas);apply_city_attendant(data,atlas);apply_crossing_queue_cues(data,atlas)
            data['metrics']=validate(data)
            path=args.output/f'{kind}-{count}.json';path.write_text(json.dumps(data,separators=(',',':')));metrics[path.name]=data['metrics'];print(path,data['metrics'],flush=True)
    (args.output/'navigation-metrics.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
