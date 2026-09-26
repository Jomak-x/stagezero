"""Actual WebGL scene captures and a 30 fps validation orbit; no image synthesis."""
import sys, json, time, math, argparse, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
import viser
from preview import load_recording
from ardy.viz.viser_utils import Character
from object_scene import ObjectSceneLayer
from scene_objects import evaluate_objects
from scene_composition import validate_scene
parser=argparse.ArgumentParser()
parser.add_argument('--recording',default=str(ROOT/'assets/recorded_g1.csv'))
parser.add_argument('--scene',required=True);parser.add_argument('--name',required=True)
parser.add_argument('--targets',action='store_true')
parser.add_argument('--port',type=int,default=24891);parser.add_argument('--video',action='store_true')
args=parser.parse_args()
out=ROOT/'review/scene-integration';out.mkdir(exist_ok=True)
doc=validate_scene(json.loads(Path(args.scene).read_text()))
server=viser.ViserServer(host='127.0.0.1',port=args.port,label='Scene visual acceptance')
server.scene.set_up_direction('+y');server.scene.world_axes.visible=False
server.scene.configure_environment_map(None);server.scene.configure_default_lights(enabled=True,cast_shadow=True)
server.scene.add_light_ambient('/fill',color=(210,225,242),intensity=.8)
from scene_ground import has_authored_ground
server.scene.add_box('/floor',color=(53,66,73),dimensions=(200,.1,200),position=(0,-.07,0),visible=not has_authored_ground(doc['objects']))
torch.set_num_threads(2)
skeleton,positions,rotations=load_recording(Path(args.recording))
character=Character('actor',server,skeleton,create_skeleton_mesh=False,create_skinned_mesh=True,mesh_mode='g1_stl',show_foot_contacts=False)
for h in character.g1_mesh_rig.mesh_handles:h.color=(206,226,233)
character.set_pose(positions[0],rotations[0])
layer=ObjectSceneLayer(server)
states=evaluate_objects(doc['objects'],positions[:1].numpy(),0,hand_indices=(25,33))
layer.update(doc['objects'],{'objects':states,'assets':doc.get('assets',[]),'effects':doc['effects'],'lighting':doc['lighting'],'targets':doc.get('targets',[]) if args.targets else [],'seconds':0})
print('WAITING FOR RENDER CLIENT',args.port,flush=True)
while not server.get_clients():time.sleep(.2)
client=next(iter(server.get_clients().values()));client.camera.up_direction=(0,1,0)
client.camera.near=.05;client.camera.far=400
camera=doc.get('camera',{'position':[4,3,10.5],'look_at':[0,2,-3.2]})
base=np.array(camera['position'],float);target=np.array(camera['look_at'],float)
for label,offset in [('front',np.array([0,0,0])),('left',np.array([-5,.3,1])),('right',np.array([2,.2,1]))]:
 client.camera.position=tuple(base+offset);client.camera.look_at=tuple(target);client.camera.fov=math.radians(48)
 server.flush();Image.fromarray(client.get_render(height=720,width=1280)).save(out/f'{args.name}-{label}.png')
 print('CAPTURED',label,flush=True)
if args.video:
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{args.name}-orbit.mp4')],stdin=subprocess.PIPE)
 times=[]
 for i in range(180):
  angle=(i/179-.5)*.3;delta=base-target;c,s=math.cos(angle),math.sin(angle)
  client.camera.position=tuple(target+np.array([c*delta[0]+s*delta[2],delta[1],-s*delta[0]+c*delta[2]]));client.camera.look_at=tuple(target)
  frame=i*2%len(positions)
  character.set_pose(positions[frame],rotations[frame])
  bundle={'objects':states,'assets':doc.get('assets',[]),'effects':doc['effects'],'lighting':doc['lighting'],'targets':doc.get('targets',[]) if args.targets else [],'seconds':i/30}
  layer.update(doc['objects'],bundle);server.flush()
  start=time.perf_counter();pixels=client.get_render(height=720,width=1280);times.append(time.perf_counter()-start)
  proc.stdin.write(pixels[:,:,:3].tobytes())
  if i%60==0:print('VIDEO',i,'/180',flush=True)
 proc.stdin.close();assert proc.wait()==0
 (out/f'{args.name}-capture-timing.json').write_text(json.dumps({'capture_roundtrip_median_ms':round(float(np.median(times))*1000,2),'note':'Offscreen render+JPEG+local websocket readback; not interactive FPS','frames':180,'video_fps':30},indent=2))
server.stop();print('DONE',args.name,flush=True)
