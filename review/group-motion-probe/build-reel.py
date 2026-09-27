from pathlib import Path
import subprocess,json,hashlib
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent; T=Path('/tmp/group-motion-reel');T.mkdir(exist_ok=True)
chapters=[
('TEN INDEPENDENT CELEBRATIONS', 'City | seed 77 | offline research capture; not live Studio support') + (R/'video'/'celebrate-ten-s77'/'playback.mp4',),
('SEQUENTIAL TIMING CONTROL', 'Same generated sources; explicit holds. Not a production planner run.') + (R/'video'/'wave-sequential-control'/'playback.mp4',),
('THREE WAVING TOGETHER', 'City | seed 71 | same sources, simultaneous schedule') + (R/'video'/'wave-three-s71'/'playback.mp4',),
('THREE WAVING / SECOND SEED', 'Market | seed 72 | different starting layout') + (R/'video'/'wave-three-s72'/'playback.mp4',),
('THREE DANCING', 'Industrial | seed 73 | modest movement, not choreography') + (R/'video'/'dance-three-s73'/'playback.mp4',),
('THREE DANCING / SECOND SEED', 'City | seed 74 | modest movement') + (R/'video'/'dance-three-s74'/'playback.mp4',),
('THREE CELEBRATING', 'Market | seed 75 | independent timing') + (R/'video'/'celebrate-three-s75'/'playback.mp4',),
('THREE CELEBRATING / SECOND SEED', 'Industrial | seed 76 | independent timing') + (R/'video'/'celebrate-three-s76'/'playback.mp4',),
('PAIR BASELINE', 'Reviewed pair arrays unchanged; authored held third observer') + (R/'video'/'pair-before-third-wave'/'playback.mp4',),
('PAIR + REQUESTED THIRD WAVE', 'Same pair arrays exactly; only third track changes') + (R/'video'/'pair-plus-third-wave'/'playback.mp4',),
]
font='/System/Library/Fonts/Supplemental/Arial.ttf'; parts=[]; records=[]; start=0

def run(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True)
def card(lines,seconds):
 global start
 i=len(parts);im=Image.new('RGB',(1280,720),'#14202b');draw=ImageDraw.Draw(im)
 for k,line in enumerate(lines):draw.text((64,210+k*65),line,font=ImageFont.truetype(font,42 if k==0 else 26),fill='white')
 png=T/f'card-{i}.png';im.save(png)
 out=T/f'{i:02}.mp4';run(['-loop','1','-framerate','30','-i',str(png),'-t',str(seconds),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out)]);parts.append(out);start+=seconds
card(['STAGEZERO / OPTIONAL GROUP EXPERIMENT','Real Core model motion; 3 and 10 independent actors.','All ten complete captures at normal speed; no internal cuts.','No production UI or pair-generation changes.'],5)
for title,subtitle,src in chapters:
 card([title,subtitle],3)
 frames=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','default=nw=1:nk=1',str(src)],text=True).strip())
 label=T/f'label-{len(parts)}.png';im=Image.new('RGBA',(1280,44),(0,0,0,165));ImageDraw.Draw(im).text((20,9),title+'  |  1x full take',font=ImageFont.truetype(font,22),fill='white');im.save(label)
 out=T/f'{len(parts):02}.mp4'
 run(['-i',str(src),'-i',str(label),'-filter_complex','[0:v]scale=1280:720,setsar=1[base];[base][1:v]overlay=0:0','-an','-r','30','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out)])
 records.append(dict(title=title,start_seconds=start,frames=frames,duration_seconds=frames/30,source=str(src.relative_to(R.parent)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
 parts.append(out);start+=frames/30
card(['LIMITS AND PRESERVED FAILURE','Mid-scene third entry: rejected; exact original take preserved.','Reason: endpoint velocity and body proportion mismatch.','Waves / celebration read best; dance is subdued; feet still rough.','Independent motion does not solve three-person physical contact.'],8)
listing=T/'concat.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in parts))
run(['-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(R/'group-comparison.mp4')])
(R/'reel-manifest.json').write_text(json.dumps({'fps':30,'chapters':records,'full_takes':True,'retimed':False,'source_cuts':False,'expected_seconds':start,'output_sha256':hashlib.sha256((R/'group-comparison.mp4').read_bytes()).hexdigest()},indent=2)+'\n')
print('Reel complete',start)
