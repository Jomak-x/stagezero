"""Encode selected real browser captures; never synthesize crowd frames."""
from pathlib import Path
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'review/crowd-demos'
selection=json.loads((BASE/'selected-captures.json').read_text())
out=BASE/'videos';out.mkdir(exist_ok=True)
def run(*args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*map(str,args)],check=True)
enc=['-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart']
segments=[]
for index,scene in enumerate(['crossing','city','station']):
    source=BASE/selection[scene]['video']
    target=out/f'{scene}-112.mp4'
    run('-i',source,'-vf','fps=30',*enc,target)
    sections={'crossing':[(22,12),(66,10),(13,8)],'city':[(40,25),(69,5)],'station':[(22,10),(43,10),(65,10)]}[scene]
    for part,(start,duration) in enumerate(sections):
        clip=out/f'highlight-{scene}-{part}.mp4'
        vf="fps=30"
        run('-ss',start,'-i',source,'-t',duration,'-vf',vf,*enc,clip);segments.append(clip)
concat=out/'highlight-concat.txt';concat.write_text(''.join(f"file '{p.name}'\n" for p in segments))
run('-f','concat','-safe','0','-i',concat,'-c','copy','-movflags','+faststart',out/'stagezero-three-worlds.mp4')
checks=[]
for file in [*(out/f'{s}-112.mp4' for s in selection),out/'stagezero-three-worlds.mp4']:
    run('-i',file,'-f','null','-')
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size:stream=codec_name,width,height,r_frame_rate','-of','json',str(file)]))
    checks.append({'file':str(file.relative_to(BASE)),'decode':'passed','sha256':hashlib.sha256(file.read_bytes()).hexdigest(),**info})
(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
