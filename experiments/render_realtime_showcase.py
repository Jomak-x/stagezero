#!/usr/bin/env python3
"""Render saved integrated scene poses, frame-for-frame at 20 fps, on CoreSkin.

Example:
  python experiments/render_realtime_showcase.py --raw scene.raw.npz \
      --report scene.report.json --output scene.mp4

The renderer never generates, interpolates, smooths, or repairs poses. Failed
quality gates stay visible. --fixture must be used for renderer test fixtures.
Geometry is captured from the same ObjectSceneLayer procedural builders used
by the studio; unsupported custom meshes are rejected rather than substituted.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor' / 'ardy'))
from ardy.skeleton import CoreSkeleton27
from ardy.viz.core_skin import CoreSkin
from interaction_scene import scene_objects
from object_scene import ObjectSceneLayer
from realtime_clip import CanonicalClip

FPS = 20
CYAN = (95, 217, 226)
GOLD = (244, 181, 102)
WHITE = (234, 241, 250)
MUTED = (153, 172, 193)
BG = (13, 20, 31)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=24)
def font(size, bold=False):
    candidates = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
                  ["/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def load_scene(raw_path, report_path, plan_path=None, *, fixture=False):
    report = json.loads(Path(report_path).read_text()) if report_path else {}
    plan = json.loads(Path(plan_path).read_text()) if plan_path else report.get('plan')
    if not isinstance(plan, dict):
        raise ValueError('A scene plan is required, directly or inside the report')
    with np.load(raw_path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].item())
        clip = CanonicalClip(archive['positions'], archive['rotations'], metadata['fps'],
                             tuple(metadata['actor_ids']), metadata.get('source', 'saved_scene'), metadata)
    if plan.get('fps', FPS) != FPS or tuple(plan['actor_ids']) != clip.actor_ids:
        raise ValueError('Plan actor order/fps does not match the saved motion')
    if plan.get('total_frames') != clip.frames:
        raise ValueError('The saved motion must contain every planned frame')
    beats = plan.get('beats')
    if not isinstance(beats, list) or not beats:
        raise ValueError('Plan must contain source-labelled beat intervals')
    cursor = 0
    for beat in beats:
        if (beat['start_frame'] != cursor or type(beat['end_frame']) is not int
                or not cursor < beat['end_frame'] <= clip.frames
                or beat['source'] not in ('ardy_core', 'intergen')):
            raise ValueError('Beat intervals must cover the clip once, in order, with known sources')
        cursor = beat['end_frame']
    if cursor != clip.frames:
        raise ValueError('Plan does not cover the complete saved clip')
    if not fixture and (report.get('status') not in ('complete', 'quality_failed', 'realtime_failed', 'offline_assembled')
                        or report.get('raw_file') != Path(raw_path).name):
        raise ValueError('A matching fully assembled scene-runner report is required; use --fixture only for a test fixture')
    shape = report.get('metrics', {}).get('shape')
    if shape is not None and shape != list(clip.positions.shape):
        raise ValueError('Report shape does not match the saved poses')
    return clip, plan, report


class _GeometryCapture:
    """Capture production procedural parts without creating a Viser server."""
    def add_box(self, name, *, dimensions, color, **kwargs):
        mesh = trimesh.creation.box(extents=dimensions)
        return SimpleNamespace(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), color=color)

    def add_icosphere(self, name, *, radius, color, subdivisions=1, **kwargs):
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        return SimpleNamespace(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), color=color)

    def add_mesh_simple(self, name, *, vertices, faces, color, **kwargs):
        return SimpleNamespace(vertices=np.asarray(vertices), faces=np.asarray(faces), color=color)


def geometry(scene):
    scene_objects(scene)  # Validate the same coordinate/size contract as planning.
    layer = ObjectSceneLayer(SimpleNamespace(scene=_GeometryCapture()))
    meshes = []
    for obj in scene['objects']:
        if obj['kind'] == 'custom':
            raise ValueError('Custom visual meshes require an explicit mesh loader; no bounding-box substitution is permitted')
        for handle, offset, _ in layer._build(obj):
            meshes.append((handle.vertices + np.asarray(offset) + np.asarray(obj['position']),
                           handle.faces, np.asarray(handle.color)))
    return meshes


def skin_cache(clip, raw_path, cache_dir):
    skeleton = CoreSkeleton27()
    skin = CoreSkin(skeleton)
    rig_path = Path(skeleton.folder) / 'skin_standard.npz'
    key = hashlib.sha256((digest(raw_path) + digest(rig_path) + 'global-CoreSkin-v1').encode()).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (key + '.vertices.npy')
    shape = (len(clip.actor_ids), clip.frames, len(skin.bind_vertices), 3)
    if target.exists():
        vertices = np.load(target, mmap_mode='r', allow_pickle=False)
        if vertices.shape == shape and vertices.dtype == np.float32 and np.isfinite(vertices).all():
            return vertices, skin.faces.numpy().astype(np.int32), key
    temp = target.with_suffix('.partial.npy')
    vertices = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=shape)
    with torch.inference_mode():
        for actor in range(len(clip.actor_ids)):
            for start in range(0, clip.frames, 16):
                stop = min(clip.frames, start + 16)
                p = torch.tensor(clip.positions[actor, start:stop])
                r = torch.tensor(clip.rotations[actor, start:stop])
                posed = skin.skin(r, p, rot_is_global=True).cpu().numpy()
                if not np.isfinite(posed).all():
                    raise ValueError('Official CoreSkin produced nonfinite vertices')
                vertices[actor, start:stop] = posed
    vertices.flush()
    del vertices
    temp.replace(target)
    return np.load(target, mmap_mode='r', allow_pickle=False), skin.faces.numpy().astype(np.int32), key


def camera(points, width, height, azimuth=30., elevation=18.):
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    az,el=math.radians(azimuth),math.radians(elevation)
    direction = np.array([math.sin(az)*math.cos(el),math.sin(el),math.cos(az)*math.cos(el)])
    forward = -direction
    right = np.cross(forward, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack((right, up, forward))
    distance = max(3., np.linalg.norm(hi-lo))
    focal = height * 1.4
    for _ in range(50):
        eye = center + direction * distance
        q = (points-eye) @ basis.T
        projected = np.abs(q[:, :2] / np.maximum(.01, q[:, 2:]) * focal)
        if q[:, 2].min() > .2 and projected[:, 0].max() < width*.43 and projected[:, 1].max() < height*.42:
            return eye, basis, center, focal
        distance *= 1.1
    raise ValueError('Could not fit the saved scene into the camera')


def raster(meshes, cam, size):
    """Offline painter raster, adapted from the existing findings renderer."""
    width, height = size
    eye, basis, center, focal = cam
    image = Image.new('RGB', size, (24, 35, 50))
    draw = ImageDraw.Draw(image)

    def project(v):
        q = (v-eye) @ basis.T
        xy = q[:, :2] / np.maximum(q[:, 2:], .01) * focal
        xy[:, 0] += width/2
        xy[:, 1] = height/2 - xy[:, 1]
        return xy, q[:, 2]

    extent = max(8, int(np.ceil(np.linalg.norm(eye-center))))
    cx, cz = int(center[0]), int(center[2])
    for n in range(-extent, extent+1):
        for ends in (([cx+n,0,cz-extent],[cx+n,0,cz+extent]),
                     ([cx-extent,0,cz+n],[cx+extent,0,cz+n])):
            xy, z = project(np.asarray(ends, dtype=float))
            if np.all(z > .1):
                draw.line([tuple(p) for p in xy], fill=(39, 53, 70), width=1)
    triangles, colors = [], []
    light = np.array([-.35,.9,-.25]); light /= np.linalg.norm(light)
    for vertices, faces, color in meshes:
        faces = np.asarray(faces)
        tri = np.asarray(vertices)[faces]
        normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:,None],1e-9)
        visible = (np.sum(normals*(eye-tri.mean(axis=1)), axis=1)>0) & (lengths>1e-10)
        if len(faces)>10000:
            vn = np.zeros_like(vertices)
            for column in range(3):
                np.add.at(vn, faces[:,column], normals)
            vn /= np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-9)
            normals = vn[faces].mean(axis=1)
            normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-9)
        shade = .4 + .6*np.maximum(0,normals[visible]@light)
        triangles.append(tri[visible])
        colors.append(np.clip(np.asarray(color)[None,:]*shade[:,None],0,255).astype(np.uint8))
    tri = np.concatenate(triangles); colors = np.concatenate(colors)
    xy,z = project(tri.reshape(-1,3)); xy=xy.reshape(-1,3,2); z=z.reshape(-1,3)
    a,b = xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]
    area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    valid=(z.min(axis=1)>.1)&(area>.001)&(xy[:,:,0].max(axis=1)>0)&(xy[:,:,0].min(axis=1)<width)&(xy[:,:,1].max(axis=1)>0)&(xy[:,:,1].min(axis=1)<height)
    order=np.flatnonzero(valid); order=order[np.argsort(z[order].mean(axis=1))[::-1]]
    for index in order:
        draw.polygon([tuple(p) for p in xy[index]], fill=tuple(colors[index]))
    return image


def wrapped(draw, text, xy, width, *, size=18, fill=WHITE, max_lines=3, bold=False):
    x,y=xy; f=font(size,bold)
    words=str(text).split(); lines=[]; current=''
    for word in words:
        attempt=(current+' '+word).strip()
        if current and draw.textlength(attempt,font=f)>width:
            lines.append(current);current=word
        else:
            current=attempt
    if current:lines.append(current)
    if len(lines)>max_lines:
        lines=lines[:max_lines];lines[-1]=lines[-1].rstrip(' .')+'...'
        while draw.textlength(lines[-1],font=f)>width and len(lines[-1])>4:
            lines[-1]=lines[-1][:-4]+'...'
    for line in lines:
        draw.text((x,y),line,font=f,fill=fill); y+=size+6
    return y


def metric_lines(report, beat):
    row=next((r for r in report.get('metrics',{}).get('beats',[]) if r['beat_id']==beat['id']),None)
    if row is None:return ['Measured gates unavailable']
    lines=[]
    # Failed checks are shown first when the sidebar has limited space.
    for gate in sorted(row['gates'],key=lambda item: bool(item['passed'])):
        req,m=gate['requirement'],gate['measurement']; label=req['metric']; prefix='PASS' if gate['passed'] else 'FAIL'
        if label=='hand_contact' or label=='object_contact':
            text=f"{m['longest_near_run_s']:.2f}s near target"
        elif label=='hand_release':text=f"final hand gap {m['mean_final_hand_gap_m']:.2f}m"
        elif label=='pair_separation':text=f"minimum root gap {m['min_root_separation_xz_m']:.2f}m"
        elif label=='scene_collision':text=f"{m['total_collision_frames']} proxy overlap frames"
        elif label=='gate_traversal':text='gate crossing proxy'
        elif label=='continuity':text=f"root boundary step {max(m['boundary_root_steps_m']):.3f}m"
        elif label=='dodge_lateral':text=f"lateral range {m['relative_lateral_range_m']:.2f}m"
        elif label=='block_guard':text=f"hand/forearm near {m['longest_near_run_s']:.2f}s"
        elif label=='push_reaction':text=f"root gap change {m['last_third_mean_root_gap_m']-m['first_third_mean_root_gap_m']:.2f}m"
        else:text=label.replace('_',' ')
        lines.append(f'{prefix}  {text}')
    return lines or ['No beat-specific gate requested']


def frame_image(frame, clip, vertices, faces, static, cam, plan, report, *, size, fixture, detail_cameras=None):
    width,height=size
    main_width=int(width*.72); view_height=height-204
    meshes=static+[(vertices[actor,frame],faces,(CYAN,GOLD)[actor]) for actor in range(len(clip.actor_ids))]
    view=raster(meshes,cam,(main_width,view_height))
    beat=next(b for b in plan['beats'] if b['start_frame']<=frame<b['end_frame'])
    if detail_cameras and beat['id'] in detail_cameras:
        inset_width,inset_height=350,230
        inset=raster(meshes,detail_cameras[beat['id']],(inset_width,inset_height-23))
        view.paste(inset,(main_width-inset_width-12,35))
        detail_draw=ImageDraw.Draw(view)
        detail_draw.rectangle((main_width-inset_width-12,12,main_width-12,35),fill=BG)
        detail_draw.text((main_width-inset_width-4,16),'DETAIL / SAME SAVED FRAME',font=font(12,True),fill=CYAN)
        detail_draw.rectangle((main_width-inset_width-12,12,main_width-12,12+inset_height),outline=MUTED,width=1)
    image=Image.new('RGB',size,BG); image.paste(view,(22,102));draw=ImageDraw.Draw(image)
    title=plan.get('name','Saved scene').replace('_',' ').title()
    draw.text((24,16),'STAGEZERO / SAVED SCENE PLAYBACK',font=font(16,True),fill=CYAN)
    wrapped(draw,title,(24,43),width-240,size=30,bold=True,max_lines=1)
    draw.text((width-195,22),f'{frame/FPS:05.2f} / {clip.frames/FPS:.1f} s',font=font(19,True),fill=WHITE)
    draw.text((width-195,50),f'{FPS} fps | frame {frame+1}/{clip.frames}',font=font(15),fill=MUTED)
    x=main_width+44; available=width-x-22;y=106
    banner=('RENDER TEST FIXTURE' if fixture else {'quality_failed':'QUALITY GATES FAILED',
            'realtime_failed':'PLAYBACK TIMING FAILED', 'offline_assembled':'OFFLINE ASSEMBLY'}.get(report.get('status'),'SAVED MODEL OUTPUT'))
    y=wrapped(draw,banner,(x,y),available,size=16,bold=True,fill=GOLD if fixture or report.get('status')!='complete' else CYAN,max_lines=2)+12
    y=wrapped(draw,beat['id'].replace('_',' ').title(),(x,y),available,size=24,bold=True,max_lines=2)+8
    source='ARDY Core / native' if beat['source']=='ardy_core' else 'InterGen > Core27 / retargeted'
    y=wrapped(draw,source,(x,y),available,size=17,fill=CYAN if beat['source']=='ardy_core' else GOLD,max_lines=2)+12
    y=wrapped(draw,'REQUESTED BEAT',(x,y),available,size=13,bold=True,fill=MUTED,max_lines=1)
    y=wrapped(draw,beat['prompt'],(x,y),available,size=17,max_lines=4)+12
    for line in metric_lines(report,beat)[:3]:
        y=wrapped(draw,line,(x,y),available,size=16,fill=GOLD if line.startswith('FAIL') else MUTED,max_lines=2)+5
    for actor,identifier in enumerate(clip.actor_ids):
        wrapped(draw,identifier,(x,height-145+actor*23),available,size=15,fill=(CYAN,GOLD)[actor],max_lines=1)
    ty=height-77; timeline_width=width-48
    for b in plan['beats']:
        left=24+int(b['start_frame']/clip.frames*timeline_width)
        right=24+int(b['end_frame']/clip.frames*timeline_width)
        draw.rectangle((left,ty,right-2,ty+8),fill=CYAN if b['source']=='ardy_core' else GOLD)
    marker=24+int(frame/max(1,clip.frames-1)*timeline_width)
    draw.line((marker,ty-4,marker,ty+13),fill=WHITE,width=2)
    disclaimer=('FIXTURE ONLY: existing native clip, not an integrated scene result.' if fixture else
                'Exact saved frames. Geometry/contact checks are proxies; no physics or semantic-action guarantee.')
    wrapped(draw,disclaimer,(24,height-55),width-48,size=15,fill=MUTED,max_lines=1)
    footer=('No pose edits, interpolation, or transition repair. Official CoreSkin rig.'+
            (' InterGen: noncommercial research.' if any(b['source']=='intergen' for b in plan['beats']) else ''))
    wrapped(draw,footer,(24,height-31),width-48,size=13,fill=MUTED,max_lines=1)
    return image


def select_snapshots(clip, plan):
    selected = {}
    def add(frame, reason):
        selected.setdefault(int(frame), []).append(reason)
    add(0, 'first frame'); add(clip.frames-1, 'last frame')
    for beat in plan['beats']:
        start,end=beat['start_frame'],beat['end_frame']
        add(start, 'beat boundary after')
        if start:add(start-1, 'beat boundary before')
        add((start+end-1)//2, 'beat midpoint')
        # These are explicitly geometric sampling rules, not semantic success.
        if beat['source']=='intergen' and len(clip.actor_ids)==2:
            hands=clip.positions[:,start:end][:,:,[11,17]]
            distance=np.linalg.norm(hands[0,:,:,None]-hands[1,:,None,:],axis=-1)
            add(start+np.argmin(distance.min(axis=(1,2))), 'closest hand-pair frame')
            speed=np.linalg.norm(np.diff(hands,axis=1),axis=-1).max(axis=(0,2))
            if len(speed):add(start+1+np.argmax(speed), 'maximum hand-motion frame')
        target=beat.get('metadata',{}).get('native_hand_target')
        if target and 'position_xyz' in target:
            joint=10 if target.get('hand','RightHand')=='RightHand' else 16
            distance=np.linalg.norm(clip.positions[:,start:end,joint]-target['position_xyz'],axis=-1)
            add(start+np.argmin(distance.min(axis=0)), 'closest wrist-target frame')
    return dict(sorted(selected.items()))


def contact_sheets(paths, selected, plan, output, size, fixture, status, detail_beats):
    main_width=int(size[0]*.72); view_height=size[1]-204
    frames=list(selected); outputs=[]
    cell_w,cell_h=480,330
    for page_start in range(0,len(frames),12):
        chunk=frames[page_start:page_start+12]
        sheet=Image.new('RGB',(3*cell_w,90+math.ceil(len(chunk)/3)*cell_h),BG)
        draw=ImageDraw.Draw(sheet)
        label=('FIXTURE ONLY' if fixture else {'quality_failed':'QUALITY FAILED / DIAGNOSTIC',
               'realtime_failed':'PLAYBACK TIMING FAILED / DIAGNOSTIC', 'offline_assembled':'OFFLINE ASSEMBLY / FRAME REVIEW'}.get(status,'SAVED SCENE / FRAME REVIEW'))
        draw.text((18,14),label,font=font(20,True),fill=GOLD if fixture or status!='complete' else CYAN)
        draw.text((18,45),plan.get('name','Scene').replace('_',' ')+' | exact saved poses, 20 fps',font=font(18),fill=WHITE)
        for i,frame in enumerate(chunk):
            x=(i%3)*cell_w;y=90+(i//3)*cell_h
            beat=next(b for b in plan['beats'] if b['start_frame']<=frame<b['end_frame'])
            is_peak=any('closest' in reason or 'maximum' in reason for reason in selected[frame])
            bounds=(22,102,22+main_width,102+view_height)
            if is_peak and beat['id'] in detail_beats:
                bounds=(22+main_width-362,102+12,22+main_width-12,102+242)
            image=Image.open(paths[frame]).convert('RGB').crop(bounds)
            image.thumbnail((cell_w-16,250),Image.Resampling.LANCZOS)
            sheet.paste(image,(x+8,y))
            beat=next(b for b in plan['beats'] if b['start_frame']<=frame<b['end_frame'])
            draw.text((x+10,y+253),f'{frame/20:05.2f}s | frame {frame} | '+beat['id'],font=font(15,True),fill=WHITE)
            wrapped(draw,'; '.join(selected[frame]),(x+10,y+276),cell_w-20,size=13,fill=MUTED,max_lines=2)
        path=output.with_name(f'{output.stem}.contact-sheet-{page_start//12+1:02d}.png')
        sheet.save(path);outputs.append(str(path.resolve()))
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=Path(tempfile.gettempdir())/'stagezero-showcase-meshes')
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--camera-azimuth',type=float,default=30.)
    parser.add_argument('--camera-elevation',type=float,default=18.)
    parser.add_argument('--fixture',action='store_true',help='Watermark renderer-test input; never label it an integrated result')
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--no-detail-inset',action='store_true')
    parser.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    args=parser.parse_args()
    if args.width<1000 or args.height<700 or args.width%2 or args.height%2:
        parser.error('Use even dimensions of at least 1000 x 700 for readable provenance')
    if not args.preview_only and not args.ffmpeg:
        parser.error('ffmpeg was not found; supply --ffmpeg (no installation is performed)')
    if args.output.suffix.lower()!='.mp4':parser.error('--output must have an .mp4 extension')
    if not math.isfinite(args.camera_azimuth) or not -70 <= args.camera_elevation <= 70:
        parser.error('Camera angles must be finite; elevation must be between -70 and 70 degrees')
    torch.set_num_threads(2)
    clip,plan,report=load_scene(args.raw,args.report,args.plan,fixture=args.fixture)
    static=geometry(plan['scene'])
    vertices,faces,key=skin_cache(clip,args.raw,args.cache_dir)
    points=np.concatenate([clip.positions.reshape(-1,3)]+[m[0] for m in static])
    cam=camera(points,int(args.width*.72),args.height-204,args.camera_azimuth,args.camera_elevation)
    detail_cameras={}; detail_angles={}
    if not args.no_detail_inset:
        for beat in plan['beats']:
            if beat['source']=='intergen' or beat.get('kind') in ('action','transition'):
                points=clip.positions[:,beat['start_frame']:beat['end_frame']].reshape(-1,3)
                azimuth=args.camera_azimuth
                if len(clip.actor_ids)==2:
                    middle=(beat['start_frame']+beat['end_frame']-1)//2
                    delta=clip.positions[1,middle,0]-clip.positions[0,middle,0]
                    if np.linalg.norm(delta[[0,2]])>.05:
                        # View across the pair, not down their separation axis.
                        across=math.degrees(math.atan2(delta[0],delta[2]))+90
                        candidates=(across,across+180)
                        azimuth=min(candidates,key=lambda a: abs((a-args.camera_azimuth+180)%360-180))
                elif beat.get('metadata',{}).get('native_hand_target'):
                    middle=(beat['start_frame']+beat['end_frame']-1)//2
                    right=clip.positions[0,middle,8]-clip.positions[0,middle,14]
                    right[1]=0
                    if np.linalg.norm(right)>.05:
                        right=right/np.linalg.norm(right)
                        backward=-np.cross([0.,1.,0.],right)
                        side=right if beat['metadata']['native_hand_target'].get('hand')=='RightHand' else -right
                        view_direction=side+backward
                        azimuth=math.degrees(math.atan2(view_direction[0],view_direction[2]))
                detail_angles[beat['id']]=azimuth
                detail_cameras[beat['id']]=camera(points,350,207,azimuth,args.camera_elevation)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    selected=select_snapshots(clip,plan)
    preview_frames=list(selected)
    snapshots=[]; snapshot_paths={}
    ff=None;log=None
    if not args.preview_only:
        log=args.output.with_suffix('.ffmpeg.log').open('w')
        ff=subprocess.Popen([args.ffmpeg,'-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{args.width}x{args.height}',
                             '-framerate',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20',
                             '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE,stderr=log)
    try:
        for frame in (preview_frames if args.preview_only else range(clip.frames)):
            image=frame_image(frame,clip,vertices,faces,static,cam,plan,report,size=(args.width,args.height),fixture=args.fixture,detail_cameras=detail_cameras)
            if frame in preview_frames:
                target=args.output.with_name(f'{args.output.stem}.frame-{frame:04d}.png')
                image.save(target);snapshots.append(str(target.resolve()));snapshot_paths[frame]=target
            if ff is not None:ff.stdin.write(np.asarray(image).tobytes())
            if frame%40==0:print(f'Rendered frame {frame+1}/{clip.frames}',flush=True)
    finally:
        if ff is not None:
            ff.stdin.close()
            code=ff.wait()
            log.close()
            if code:raise RuntimeError(f'ffmpeg exited {code}; inspect {args.output.with_suffix(".ffmpeg.log")}')
    sheets=contact_sheets(snapshot_paths,selected,plan,args.output,(args.width,args.height),args.fixture,report.get('status'),set(detail_cameras))
    manifest={'output':None if args.preview_only else str(args.output.resolve()),'input':str(args.raw.resolve()),
              'input_sha256':digest(args.raw),'report':None if args.report is None else str(args.report.resolve()),
              'report_sha256':None if args.report is None else digest(args.report),'frames':clip.frames,'fps':FPS,
              'duration_seconds':clip.frames/FPS,'fixture':args.fixture,'source_status':report.get('status','unverified'),
              'actor_ids':clip.actor_ids,'mesh_vertices_per_actor':vertices.shape[2],'mesh_faces_per_actor':len(faces),
              'mesh_cache_key':key,'snapshots':snapshots,'snapshot_selection':selected,'contact_sheets':sheets,
              'camera':{'azimuth_degrees':args.camera_azimuth,'elevation_degrees':args.camera_elevation,
                        'same_frame_detail_inset_beats':list(detail_cameras),'detail_azimuth_degrees':detail_angles},'no_pose_edits':True,'no_interpolation':True,
              'geometry':'Exact ObjectSceneLayer procedural parts; static planned scene state',
              'renderer':'Official CoreSkin CPU LBS; PIL painter raster; ffmpeg libx264',
              'beats':[{k:b[k] for k in ('id','source','start_frame','end_frame')} for b in plan['beats']]}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
