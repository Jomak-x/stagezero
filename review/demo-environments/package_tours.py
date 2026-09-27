# Run from the repository root after capturing all three tours in the browser.
# Requires ffmpeg and ffprobe on PATH.
from pathlib import Path
import subprocess,json,hashlib,shutil
p=Path('review/demo-environments');out=p/'preview';records=[]
for scene in ['crossing','city','station']:
 src=sorted((p/'captures').glob(scene+'-camera-tour-*.webm'))[-1];dst=out/f'{scene}-tour.mp4'
 if not dst.exists() or dst.stat().st_mtime<src.stat().st_mtime:
  subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(src),'-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-fps_mode','passthrough','-movflags','+faststart',str(dst)],check=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,nb_read_frames,duration:format=duration,size','-of','json',str(dst)]))
 duration=float(probe['format']['duration']);assert 35<duration<38
 records.append({'scene':scene,'source_capture':str(src.relative_to(p)),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'mp4':str(dst.relative_to(p)),'mp4_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'probe':probe,'method':'Actual browser canvas MediaRecorder30fps,6shots×6seconds; H264transcode without retiming. Not prerendered frames.'})
 latest={}
 for f in sorted((p/'captures').glob(scene+'-*.json')):
  d=json.loads(f.read_text());latest[d['camera']['id']]=(f,d)
 for key,(f,d) in latest.items():shutil.copy2(f.with_suffix('.png'),out/f'{scene}-{key}.png')
 (out/f'{scene}-preset.json').write_text(json.dumps({'scene':scene,'cameras':[latest[k][1]['camera'] for k in {'crossing':['hero','street','centre','overhead','corner','avenue'],'city':['hero','street','cafe','bookshop','courtyard','overhead'],'station':['hero','street','canopy','hall','bus','overhead']}[scene]],'metadata':d['metadata'],'geometry_source':'studio_client/src/environments/EnvironmentScene.ts','geometry_seed':{'crossing':239,'city':827,'station':1069}[scene]},indent=2)+'\n')
(p/'video-provenance.json').write_text(json.dumps(records,indent=2)+'\n')
index=json.loads((p/'capture-index.json').read_text())
for row in index['stills']:
 src=sorted((p/'captures').glob(row['scene']+'-'+row['camera']+'-*.json'))[-1];row['capture']=str(src.relative_to(p));row['sha256']=hashlib.sha256((p/row['preview']).read_bytes()).hexdigest()
index['checks']['typescript']='pass';index['checks']['vite_production_build']='pass';index['checks']['mp4_decode_and_duration']='3 complete tours verified, approximately36seconds each';index['checks']['performance_scope']='Observed live HUD near60FPS on this Mac; not an actor-scale performance benchmark.'
(p/'capture-index.json').write_text(json.dumps(index,indent=2)+'\n')
prov=json.loads((p/'source-provenance.json').read_text())
for f in prov['code_and_assets']:prov['code_and_assets'][f]={'sha256':hashlib.sha256(Path(f).read_bytes()).hexdigest(),'bytes':Path(f).stat().st_size}
prov['capture_note']='Revised compact crossing, colorful street dressing and warm cafe. Crossing capture preceded cafe-only refinement; crossing geometry is unchanged. Selected captures supersede previous studies; raw trials remain local.'
(p/'source-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps([{ 'scene':r['scene'],'duration':r['probe']['format']['duration'],'bytes':r['probe']['format']['size']}for r in records]))

from pathlib import Path
import subprocess,json,hashlib
p=Path('review/demo-environments/preview');scenes=['crossing','city','station']
listing=Path('/private/tmp/background-concat.txt');listing.write_text(''.join("file '"+str((p/(s+'-tour.mp4')).resolve())+"'\n" for s in scenes))
out=p/'all-backgrounds-revised.mp4'
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-vf','fps=30','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height:format=duration,size','-of','json',str(out)]))
assert 105<float(probe['format']['duration'])<112
subprocess.run(['ffmpeg','-v','error','-i',str(out),'-f','null','-'],check=True)
(p.parent/'combined-video-provenance.json').write_text(json.dumps({'file':str(out.relative_to(p.parent)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'chapters':[{'scene':s,'starts_at_seconds':i*36} for i,s in enumerate(scenes)],'method':'Concatenated actual browser camera tours. No generated or synthetic people, no interpolation or speed changes; timestamps normalized to 30fps at concatenation. Chapter starts approximate to a frame.','probe':probe},indent=2)+'\n')
print(json.dumps(probe))
