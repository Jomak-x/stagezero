from pathlib import Path
import subprocess,json,hashlib
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent; T=Path('/tmp/group-integration-reel');T.mkdir(exist_ok=True)
chapters=[
('PROMPT: THREE CELEBRATE', 'City | seed 91 | fresh planner + fresh Core') + (R/'video'/'celebrate-city-91'/'playback.mp4',),
('PROMPT: THREE WAVE', 'Market | seed 92 | fresh planner + fresh Core') + (R/'video'/'wave-market-92'/'playback.mp4',),
('PROMPT: THREE DANCE', 'Industrial | seed 93 | modest independent dancing') + (R/'video'/'dance-industrial-93'/'playback.mp4',),
('PROMPT: HANDSHAKE WHILE THIRD WAVES', 'City | seed 94 | pair unchanged; source crouching remains') + (R/'video'/'pair-wave-city-94'/'playback.mp4',),
('GENERATED THROUGH THE STUDIO INPUT', 'Market | seed 42 | auto placement; playback, save and export verified') + (R/'video'/'ui-market-celebrate-42'/'playback.mp4',),
]
font='/System/Library/Fonts/Supplemental/Arial.ttf'; parts=[]; records=[]; start=0

def run(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True)
def card(lines,seconds):
 global start
 i=len(parts);im=Image.new('RGB',(1280,720),'#14202b');draw=ImageDraw.Draw(im)
 for k,line in enumerate(lines):draw.text((64,210+k*65),line,font=ImageFont.truetype(font,42 if k==0 else 26),fill='white')
 png=T/f'card-{i}.png';im.save(png)
 out=T/f'{i:02}.mp4';run(['-loop','1','-framerate','30','-i',str(png),'-t',str(seconds),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out)]);parts.append(out);start+=seconds
card(['STAGEZERO / INPUT-DRIVEN GROUP MOTION','Five complete fresh performances at normal speed.','Existing prompt box, latest main UI, no new mode.','Production: 1-3 actors. Ten actors remain research-only.'],5)
for title,subtitle,src in chapters:
 card([title,subtitle],3)
 frames=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','default=nw=1:nk=1',str(src)],text=True).strip())
 label=T/f'label-{len(parts)}.png';im=Image.new('RGBA',(1280,44),(0,0,0,165));ImageDraw.Draw(im).text((20,9),title+'  |  1x full take',font=ImageFont.truetype(font,22),fill='white');im.save(label)
 out=T/f'{len(parts):02}.mp4'
 run(['-i',str(src),'-i',str(label),'-filter_complex','[0:v]scale=1280:720,setsar=1[base];[base][1:v]overlay=0:0','-an','-r','30','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out)])
 records.append(dict(title=title,start_seconds=start,frames=frames,duration_seconds=frames/30,source=str(src.relative_to(R.parent)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
 parts.append(out);start+=frames/30
card(['VERIFIED AND BOUNDED','Final code reproduces every retained model take byte for byte.','Pair tracks unchanged. Failed overlays preserve original motion.','Feet and pair contact remain imperfect; no joint three-body contact.'],6)
listing=T/'concat.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in parts))
run(['-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(R/'prompt-demo.mp4')])
(R/'reel-manifest.json').write_text(json.dumps({'fps':30,'chapters':records,'full_takes':True,'retimed':False,'source_cuts':False,'expected_seconds':start,'output_sha256':hashlib.sha256((R/'prompt-demo.mp4').read_bytes()).hexdigest()},indent=2)+'\n')
print('Reel complete',start)
