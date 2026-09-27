"""Summarize hash-matched actual browser measurements and saved crowd evidence."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'review/crowd-demos'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(d):
    for scene in ['crossing','city','station']:
        key=f'/review/crowd-demos/{scene}-{d["count"]}.json'
        if scene==d['scene']:assert d['inputHashes'][key]['sha256']==sha(BASE/f'{scene}-{d["count"]}.json')
    assert d['inputHashes']['/review/crowd-demos/assets/manifest.json']['sha256']==sha(BASE/'assets/manifest.json')
    assert d['validForegroundRun'] and d['positionParity']['passed']
def row(d):
    return {'scene':d['scene'],'count':d['count'],'seconds':d['durationWallSeconds'],'fps':d['sustainedFps'],'frameMs':d['frameMs'],'gpuMs':d['gpuMs'],'animationCpuMs':d['animationCpuMs'],'renderSubmissionCpuMs':d['renderSubmissionCpuMs'],'loadMs':d['loadMs'],'moduleToReadyMs':d['moduleToReadyMs'],'heapPeakMB':d['heapBytes']['max']/1e6,'slowFramesOver50ms':len(d['slowFrames']),'device':d['device']}
selected=read(BASE/'selected-captures.json');assert set(selected)=={'crossing','city','station'}
films=[]
for scene,s in selected.items():
    d=read(BASE/s['profile']);check(d);assert (BASE/s['video']).exists();films.append(dict(profile=s['profile'],video=s['video'],**row(d)))
bench=[]
for scene,counts in [('crossing',[16,32,64,100,128]),('city',[128]),('station',[128])]:
    for count in counts:
        files=sorted((BASE/'browser').glob(f'{scene}-{count}-view*.json'))
        valid=[]
        for p in files:
            d=read(p)
            if d.get('schema')!='stagezero.saved-crowd-profile.v1' or d.get('capture'):continue
            try:check(d)
            except (AssertionError,KeyError):continue
            valid.append((p,d))
        assert valid,f'Missing hash-matched benchmark {scene} {count}'
        p,d=valid[-1];bench.append(dict(profile=str(p.relative_to(BASE)),**row(d)))
summary={'films':films,'benchmarks':bench,'measurementScope':'Actual foreground Chrome, warmed cameras, shared development machine; see RESULTS for concurrent offline planning. Heap excludes GPU and total process memory.'}
(BASE/'browser-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=['crowd_demo_navigation.py','crowd_demo_layout.py','crowd_demo_cues.py','crowd_demo_motion_export.py','crowd_demo_gait_metrics.py','studio_client/src/demos/main.ts','studio_client/src/demos/DemoCrowdRenderer.ts','studio_client/src/crowd/CrowdRenderer.ts','studio_client/src/environments/EnvironmentScene.ts','studio_client/src/environments/StreetDressing.ts','experiments/build_crowd_demos.py','experiments/crowd_demo_motions.py']
evidence={'baseCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{f:{'sha256':sha(ROOT/f),'bytes':(ROOT/f).stat().st_size} for f in files}}
(BASE/'source-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
for group in ['films','benchmarks']:
 print(group)
 for r in summary[group]:print(f"{r['scene']} {r['count']}: {r['fps']:.1f}FPS p95 {r['frameMs']['p95']:.1f}ms max {r['frameMs']['max']:.1f}ms load {r['loadMs']:.0f}ms heap {r['heapPeakMB']:.1f}MB")
