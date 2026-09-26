"""Play native InterGen paired 22-joint output without retargeting or pose edits."""
from pathlib import Path
import argparse
import json
import threading
import time
import numpy as np

CHAINS = ((0,2,5,8,11),(0,1,4,7,10),(0,3,6,9,12,15),(9,14,17,19,21),(9,13,16,18,20))
EDGES = np.array([(a,b) for chain in CHAINS for a,b in zip(chain,chain[1:])])

def serve(directory, port):
    import viser
    paths = sorted(directory.glob('*.npz'))
    if not paths:
        raise ValueError('No InterGen archives found')
    def read(path):
        with np.load(path,allow_pickle=False) as a:
            joints = a['joints'].copy()
            meta = json.loads(str(a['metadata'].item()))
        if joints.ndim != 4 or joints.shape[1:] != (2,22,3) or not np.isfinite(joints).all():
            raise ValueError('Expected finite native joints[T,2,22,3]')
        return joints, meta
    poses, meta = read(paths[0])
    server = viser.ViserServer(host='127.0.0.1',port=port,label='InterGen paired acting')
    if server.get_port() != port:
        server.stop(); raise OSError('Requested port is occupied')
    server.scene.set_up_direction('+y')
    server.scene.world_axes.visible=False
    server.gui.configure_theme(dark_mode=True,show_logo=False,show_share_button=False)
    server.scene.add_box('/floor',dimensions=(30,.04,30),position=(0,-.06,0),color=(32,40,55))
    server.gui.add_markdown('# InterGen paired acting\nNative 22-joint model output · research preview')
    choice=server.gui.add_dropdown('Take',[p.name for p in paths],initial_value=paths[0].name)
    caption=server.gui.add_markdown('')
    play=server.gui.add_button('Play')
    pause=server.gui.add_button('Pause')
    reset=server.gui.add_button('Reset view')
    slider=server.gui.add_slider('Frame',min=0,max=len(poses)-1,step=1,initial_value=0)
    state={'poses':poses,'playing':False,'started':0.,'frame':0}
    lock=threading.RLock()
    handles=[]
    for i,c in enumerate(((61,212,219),(244,180,68))):
        lines=server.scene.add_line_segments(f'/actor{i}/bones',points=poses[0,i][EDGES],colors=c,line_width=5)
        joints=server.scene.add_point_cloud(f'/actor{i}/joints',points=poses[0,i],colors=c,point_size=.055,point_shape='circle')
        handles.append((lines,joints))
    def show(frame):
        state['frame']=frame
        for i,(lines,joints) in enumerate(handles):
            lines.points=state['poses'][frame,i][EDGES]
            joints.points=state['poses'][frame,i]
        slider.value=frame
    def describe(meta):
        caption.content=f"**Prompt:** {meta.get('prompt',meta.get('caption','See archive'))}\n\n30 fps · two people jointly generated in shared coordinates. No added pair offsets or smoothing. Skeletons show native model joints; they have not been retargeted to the studio rig."
    describe(meta)
    def camera(client):
        pts=state['poses'].reshape(-1,3)
        center=(pts.min(0)+pts.max(0))/2
        client.camera.position=center+np.array([2.5,1.6,3.])
        client.camera.look_at=center
    @server.on_client_connect
    def connected(client):
        camera(client)
    @reset.on_click
    def reset_view(_):
        for client in server.get_clients().values(): camera(client)
    @slider.on_update
    def scrub(event):
        if event.client is None:
            return
        with lock:
            show(slider.value)
            state['started']=time.perf_counter()-slider.value/30
    @play.on_click
    def begin(_):
        with lock:
            if state['frame']>=len(state['poses'])-1: show(0)
            state['playing']=True
            state['started']=time.perf_counter()-state['frame']/30
    @pause.on_click
    def stop(_):
        state['playing']=False
    @choice.on_update
    def change(_):
        with lock:
            state['playing']=False
            state['poses'],meta=read(directory/choice.value)
            slider.max=len(state['poses'])-1
            show(0); describe(meta)
    print(f'InterGen viewer http://127.0.0.1:{port}/',flush=True)
    try:
        while True:
            with lock:
                if state['playing']:
                    frame=min(int((time.perf_counter()-state['started'])*30),len(state['poses'])-1)
                    if frame != state['frame']: show(frame)
                    if frame==len(state['poses'])-1: state['playing']=False
            time.sleep(1/60)
    finally:
        server.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--port',type=int,default=2343)
    a=p.parse_args(); serve(a.input,a.port)
