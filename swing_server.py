#!/usr/bin/env python3
"""Isolated live swing experiment. Physics is immediate; Core refreshes body motion asynchronously."""
from __future__ import annotations
import argparse, hashlib, json, math, mimetypes, queue, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/ardy'))

def plain(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

def rig_data():
    from ardy.skeleton import CoreSkeleton27
    from ardy.viz.core_skin import CoreSkin
    skin=CoreSkin(CoreSkeleton27())
    return {k:getattr(skin,k).numpy().tolist() for k in ('bind_vertices','faces','lbs_indices','lbs_weights','bind_rig_transform_inv')}

class LiveRun:
    def __init__(self,args):
        from swing_scene import load_swing_scene
        from swing_dynamics import SwingController
        from swing_pose import PoseController
        self.args=args; self.scene=load_swing_scene();self.controller_class=SwingController;self.pose_class=PoseController
        self.lock=threading.RLock();self.requests=queue.Queue(maxsize=2);self.stop=threading.Event()
        self.rig=rig_data();self.run_id='';self.reset()
        threading.Thread(target=self.loop,daemon=True).start()
        threading.Thread(target=self.model_loop,daemon=True).start()
    def reset(self):
        self.controller=self.controller_class(self.scene)
        self.pose=self.pose_class(ROOT/'assets/swing-motion/swing.npz')
        self.frame=0;self.time=0.;self.command_id=0;self.commands=[];self.events=[];self.last_pose=None
        self.metrics={'latency_ms':None,'stalls':0,'collisions':0,'grip':0.,'model_refreshes':0,'model_failures':0,'max_pose_step_m':0.}
        self.run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        self.run_dir=self.args.output/self.run_id;self.run_dir.mkdir(parents=True,exist_ok=True)
        self.log=(self.run_dir/'frames.jsonl').open('w');self.created=time.monotonic();self.last_command_id=0
        self.state={};self.latest_native_epoch=0;self.recording_sealed=False
        self.event('reset','New continuous run')
    def event(self,kind,text,**extra):
        self.events.append({'time':round(self.time,3),'kind':kind,'text':text,**extra});self.events=self.events[-30:]
    def command(self,action):
        with self.lock:
            if action=='reset':
                self.save();self.log.close();self.reset();return {'ok':True,'run_id':self.run_id}
            aliases={}
            kind=aliases.get(action,action)
            self.command_id+=1;cid=self.command_id
            accepted=self.controller.command(kind,timestamp=time.monotonic())
            command={'id':cid,'action':action,'time':self.time,'received_monotonic':time.monotonic(),'result':plain(accepted)}
            self.commands.append(command);self.last_command_id=cid
            self.event('command',action,command_id=cid)
            prompt={'left':'A person leans to the left and reaches overhead with the left hand.',
                    'right':'A person leans to the right and reaches overhead with the left hand.',
                    'carry':'A person carefully supports another adult on their back and balances.',
                    'land':'A person bends the knees to absorb a landing and stands steadily.',
                    'kiss':'A person stands close to a partner and gently leans the head forward.'}.get(action,'A person reaches one arm overhead and balances.')
            try:self.requests.put_nowait((self.run_id,cid,prompt))
            except queue.Full:self.event('model','Native refresh queue full; constrained body playback continues')
            return {'ok':True,'command_id':cid,'result':plain(accepted)}
    def telemetry(self,data):
        with self.lock:
            cid=data.get('command_id')
            for c in self.commands:
                if c['id']==cid and 'visible_latency_ms' not in c:
                    c['visible_latency_ms']=(time.monotonic()-c['received_monotonic'])*1000
                    c['visible_frame']=data.get('frame');self.metrics['latency_ms']=c['visible_latency_ms']
            if isinstance(data.get('stall_count'),(int,float)):self.metrics['browser_stalls']=data['stall_count']
            if isinstance(data.get('client_frame_ms'),(int,float)):self.metrics['client_frame_ms']=data['client_frame_ms']
    def loop(self):
        step=1/60;deadline=time.monotonic()
        while not self.stop.is_set():
            with self.lock:
                try:
                    self.controller.step(step);snap=self.controller.snapshot()
                    p=self.pose.get_pose(snap,self.time)
                    if hasattr(self.controller,'validate_web'):
                        self.controller.validate_web(p['web_hand'])
                        snap=self.controller.snapshot()
                    positions=np.asarray(p['positions']);rot=np.asarray(p['rotations'])
                    if positions.shape!=(2,27,3) or rot.shape!=(2,27,3,3) or not np.isfinite(positions).all() or not np.isfinite(rot).all():raise ValueError('Invalid pose')
                    if self.last_pose is not None:self.metrics['max_pose_step_m']=max(self.metrics['max_pose_step_m'],float(np.linalg.norm(positions-self.last_pose,axis=-1).max()))
                    self.last_pose=positions.copy();self.time+=step;self.frame+=1
                    pose_metrics=p.get('metrics',p.get('grip_metrics',{}));self.metrics.update({k:plain(v) for k,v in snap.get('metrics',{}).items()})
                    self.state={**plain(snap),'time':self.time,'frame':self.frame,'run_id':self.run_id,'recording_sealed':self.recording_sealed,'last_command_id':self.last_command_id,
                      'web_anchor':snap.get('anchor',{}).get('position') if snap.get('anchor') else None,'web_hand':plain(p.get('web_hand')),'actors':[{'id':name,'positions':positions[i].tolist(),'rotations':rot[i].tolist(),'scale':float(p.get('actor_scales',[1,1])[i]),'face':plain(p.get('face',p.get('face_factors',{})))} for i,name in enumerate(('spider','mj'))],
                      'metrics':{**self.metrics,'pose':plain(pose_metrics)},'controller_events':plain(snap.get('events',[])),'events':self.events[-8:],'pose_provenance':plain(p.get('provenance',{}))}
                    if not self.recording_sealed and self.command_id:
                        self.log.write(json.dumps(self.state,separators=(',',':'),allow_nan=False)+'\n')
                    if self.frame%60==0:self.log.flush()
                except Exception as exc:
                    self.state={'error':str(exc),'events':self.events,'frame':self.frame};self.event('error',str(exc));self.stop.set();print('SIMULATION ERROR',repr(exc),flush=True)
            deadline+=step;remaining=deadline-time.monotonic()
            if remaining>0:self.stop.wait(remaining)
            elif remaining<-.1:self.metrics['stalls']+=1;deadline=time.monotonic()
    def model_loop(self):
        if not self.args.token_file:return
        from realtime_client import RealtimeClient
        client=RealtimeClient(self.args.backend,self.args.token_file.read_text().strip(),job_timeout=120)
        while not self.stop.is_set():
            try:run,cid,prompt=self.requests.get(timeout=.5)
            except queue.Empty:continue
            start=time.monotonic()
            try:
                chunks=client.wait({'request_id':'swing-'+uuid.uuid4().hex,'stage_kind':'approach','frames':40,'prompt':prompt,'actor_ids':['spider'],'seed':42+cid})
                pos=np.concatenate([c.positions for c in chunks],axis=1);rot=np.concatenate([c.rotations for c in chunks],axis=1)
                with self.lock:
                    if run!=self.run_id:continue
                    metadata={'model':'ARDY Core','prompt':prompt,'command_id':cid,'response_seconds':time.monotonic()-start}
                    np.savez_compressed(self.run_dir/f'core-command-{cid}.npz',positions=pos,rotations=rot,metadata=json.dumps(metadata))
                    self.pose.set_clip(pos,rot,metadata)
                    self.metrics['model_refreshes']+=1;self.metrics['last_model_response_seconds']=metadata['response_seconds'];self.event('model','Native Core body refresh applied',command_id=cid)
            except Exception as exc:
                with self.lock:self.metrics['model_failures']+=1;self.event('model_error',str(exc))
    def save(self, recording=False):
        self.log.flush()
        report={'run_id':self.run_id,'duration_seconds':self.time,'frames':self.frame,'commands':self.commands,'metrics':self.metrics,'events':self.events,'controller':plain(self.controller.snapshot()),'scene_provenance':self.scene.get('provenance',self.scene.get('source',{})),'limitations':['Hybrid constrained experiment; flight and carry constraints are authored, not learned physical skills.','Collision proxies and measured joint/mesh checks are not rigid-body physical contact.']}
        payload=json.dumps(plain(report),indent=2,allow_nan=False)
        target='report.json' if recording or not (self.run_dir/'recorded-report.json').exists() else 'latest-state-report.json'
        (self.run_dir/target).write_text(payload)
        if recording:
            (self.run_dir/'recorded-report.json').write_text(payload)
            self.recording_sealed=True
        return self.run_dir

class Server(ThreadingHTTPServer):daemon_threads=True
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,data,kind='application/json'):
        body=json.dumps(plain(data),allow_nan=False).encode() if kind=='application/json' else data
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path;run=self.server.run
        try:
            if path=='/api/scene':return self.send(200,run.scene)
            if path=='/api/rig':return self.send(200,run.rig)
            if path=='/api/state':
                with run.lock:return self.send(200,run.state)
            if path.startswith('/vendor/three/'):
                name=Path(path).name
                if name not in ('three.module.js','three.core.js'):return self.send(404,{'error':'Not found'})
                target=self.server.three/name
            else:
                target=(ROOT/'swing_web'/('index.html' if path=='/' else path.removeprefix('/swing_web/').lstrip('/'))).resolve()
                if not target.is_relative_to(ROOT/'swing_web'):return self.send(403,{'error':'Forbidden'})
            if not target.is_file():return self.send(404,{'error':'Not found'})
            return self.send(200,target.read_bytes(),mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as exc:self.send(500,{'error':str(exc)})
    def do_POST(self):
        path=urlparse(self.path);run=self.server.run
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<0 or n>256*1024*1024:return self.send(413,{'error':'Body too large'})
            body=self.rfile.read(n)
            if path.path=='/api/recording':
                if not body:return self.send(400,{'error':'Empty recording'})
                if run.recording_sealed:return self.send(409,{'error':'Reset before recording another measured take'})
                with run.lock:
                    out=run.save(recording=True)/'interactive-run.webm';out.write_bytes(body)
                return self.send(200,{'path':str(out),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
            if n>16384:return self.send(413,{'error':'JSON body too large'})
            data=json.loads(body or b'{}')
            if path.path=='/api/command':return self.send(200,run.command(data['action']))
            if path.path=='/api/telemetry':run.telemetry(data);return self.send(200,{'ok':True})
            if path.path=='/api/save':
                with run.lock:return self.send(200,{'path':str(run.save())})
            return self.send(404,{'error':'Not found'})
        except Exception as exc:self.send(400,{'error':str(exc)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=2360);p.add_argument('--output',type=Path,default=ROOT/'.runtime/swing');p.add_argument('--token-file',type=Path);p.add_argument('--backend',default='http://127.0.0.1:8769');p.add_argument('--three-dir',type=Path,default=ROOT/'studio_client/node_modules/three/build');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);run=LiveRun(a);server=Server(('127.0.0.1',a.port),Handler);server.run=run;server.three=a.three_dir
    print(f'Live swing http://127.0.0.1:{a.port} run {run.run_id}',flush=True)
    try:server.serve_forever()
    finally:run.stop.set();run.save();server.server_close()
if __name__=='__main__':main()
