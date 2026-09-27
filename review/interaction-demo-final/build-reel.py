from pathlib import Path
import subprocess,json,hashlib
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent; T=Path('/tmp/interaction-reel');T.mkdir(exist_ok=True)
chapters=[
('01  LONG PARTNER CHANGE','City | seed 62 | 3 actors | two 7-second greetings',R/'video/city-long-supported/playback.mp4'),
('02  SPAR THEN GREET','Industrial | seed 63 | wide starts | close-contact roughness remains',R/'video/industrial-spar-greet/playback.mp4'),
('03  PARTNER DANCE','Market | seed 64 | main-compatible recovery | close-contact roughness',R/'video/market-dance/playback.mp4'),
('04  SOLO REGRESSION','City | seed 65 | walk, turn and wave | 8 seconds',R/'video/city-solo-wave/playback.mp4'),
('05  OBSERVER ATTENTION','Market | seed 48 | previously reviewed reference | unchanged after fix',R.parent/'interaction-v2/video/warmup-market48/playback.mp4')]
font='/System/Library/Fonts/Supplemental/Arial.ttf'; parts=[]; records=[]; start=0

def run(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True)
def card(lines,seconds):
 global start
 i=len(parts);im=Image.new('RGB',(1280,720),'#14202b');draw=ImageDraw.Draw(im)
 for k,line in enumerate(lines):draw.text((64,210+k*65),line,font=ImageFont.truetype(font,42 if k==0 else 26),fill='white')
 png=T/f'card-{i}.png';im.save(png)
 out=T/f'{i:02}.mp4';run(['-loop','1','-framerate','30','-i',str(png),'-t',str(seconds),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out)]);parts.append(out);start+=seconds
card(['STAGEZERO  /  FINAL DEMO REVIEW','Five complete performances at normal speed.','Four fresh heavier tests + one observer reference.','Real Core / InterGen motion; frozen plans; no internal cuts.'],5)
for title,subtitle,src in chapters:
 card([title,subtitle],3)
 frames=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','default=nw=1:nk=1',str(src)],text=True).strip())
 label=T/f'label-{len(parts)}.png';im=Image.new('RGBA',(1280,44),(0,0,0,165));ImageDraw.Draw(im).text((20,9),title+'  |  1x full take',font=ImageFont.truetype(font,22),fill='white');im.save(label)
 out=T/f'{len(parts):02}.mp4'
 run(['-i',str(src),'-i',str(label),'-filter_complex','[0:v]scale=1280:720,setsar=1[base];[base][1:v]overlay=0:0','-an','-r','30','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(out)])
 records.append(dict(title=title,start_seconds=start,frames=frames,duration_seconds=frames/30,source=str(src.relative_to(R.parent)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
 parts.append(out);start+=frames/30
card(['BOUNDARIES FOUND, NOT HIDDEN','3-beat seed 61 and 4-beat seed 66 rejected on candidate AND main.','Invalid City start corrected on the same seed; original failure retained.','Dance recovered using the original arrival policy; both attempts saved.','Contact and foot sliding remain imperfect. See the full evidence.'],8)
listing=T/'concat.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in parts))
run(['-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(R/'demo-review.mp4')])
(R/'reel-manifest.json').write_text(json.dumps({'fps':30,'chapters':records,'full_takes':True,'retimed':False,'source_cuts':False,'expected_seconds':start,'output_sha256':hashlib.sha256((R/'demo-review.mp4').read_bytes()).hexdigest()},indent=2)+'\n')
print('Reel complete',start)
