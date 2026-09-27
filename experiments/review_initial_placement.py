"""Render live Neon staging with recorded motion, never fresh-motion evidence.

Requires numpy, requests, Pillow and ffmpeg. Credentials are read privately from
--gateway-env (optional); only plans, resolved marks and source hashes are saved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cast_observer_motion import PARENTS
from object_generation import GatewayGenerator
from prompt_scene_plan import ScenePromptPlanner, auto_place
from prompt_scene_builder import place_initial_pose
from native_pair_transition import shared_place_pair

SCENE = {'version': 2, 'name': 'Clear review stage', 'objects': [], 'effects': [], 'lighting': 'neutral'}
CASES = [
    ('One character / explicit mark and facing',
     'One person named Alice starts at x -2 z 1 facing +X (90 degrees) and waves hello.'),
    ('Two characters / independent directions',
     'Alice starts at x -2 z 0 facing +X (90 degrees). Bob starts at x 2 z 0 facing -X (-90 degrees). Alice waves, then Bob waves. They remain at their separate marks.'),
    ('Three characters / automatic staging',
     'Three people named Alice, Bob and Carol wave hello in turn. Choose sensible clear starting positions and initial facing directions for all three people.')]
COLORS = [(62, 218, 219), (255, 184, 82), (191, 147, 255)]


def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render(records, source, destination):
    width, height, fps = 1280, 720, 30
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    thumbnails = []
    try:
        for record in records:
            tracks = []
            for actor in record['plan']['actors']:
                start = record['placement']['starts'][actor['id']]
                placed, yaw, offset = place_initial_pose(source[0], start)
                track = shared_place_pair(np.repeat(source[:, None], 2, axis=1), yaw=yaw, translation=offset)[:, 0]
                np.testing.assert_allclose(track[0], placed)
                tracks.append(track)
            def project(p):
                x, y, z = p
                return (round(390+48*x-26*z), round(400+13*x+22*z-90*y))
            for frame in range(180):
                im = Image.new('RGB', (width, height), '#101823'); d = ImageDraw.Draw(im)
                d.text((36, 24), 'NEON  /  CHARACTER STARTS', font=font(28), fill='#f2f7ff')
                d.text((36, 65), record['title'], font=font(22), fill='#bacbdc')
                d.rounded_rectangle((26, 113, 792, 591), radius=15, fill='#192737')
                for k in range(-5, 6):
                    d.line([project((k, 0, -5)), project((k, 0, 5))], fill='#334455', width=1)
                    d.line([project((-5, 0, k)), project((5, 0, k))], fill='#334455', width=1)
                d.text((45, 130), '3D placement · recorded pose/motion replay', font=font(17), fill='#a9bbce')
                d.text((823, 124), 'INITIAL MARKS', font=font(21), fill='#f2f7ff')
                d.text((823, 157), 'World yaw: 0° +Z / 90° +X', font=font(16), fill='#a9bbce')
                # Hold the exact first pose for two seconds, then replay 4 seconds.
                index = max(0, min(len(source)-1, frame-60))
                for i, (actor, track) in enumerate(zip(record['plan']['actors'], tracks)):
                    start = record['placement']['starts'][actor['id']]; color = COLORS[i]
                    point = np.array([start['x'], 0., start['z']]); angle = math.radians(start['yaw_degrees'])
                    end = point+np.array([math.sin(angle), 0., math.cos(angle)])*1.2
                    d.line([project(point), project(end)], fill=color, width=5)
                    for side in (-1, 1):
                        tip = end-np.array([math.sin(angle+side*.45), 0., math.cos(angle+side*.45)])*.3
                        d.line([project(end), project(tip)], fill=color, width=4)
                    pose = track[index]
                    for joint, parent in enumerate(PARENTS):
                        if parent >= 0:
                            d.line([project(pose[parent]), project(pose[joint])], fill=color, width=7)
                    head = project(pose[15]); d.ellipse((head[0]-9, head[1]-9, head[0]+9, head[1]+9), fill=color)
                    y = 212+i*105
                    d.text((823,y), actor['name'], font=font(24), fill=color)
                    d.text((823,y+34), f"x {start['x']:+.1f} m   z {start['z']:+.1f} m", font=font(19), fill='#e3eaf3')
                    d.text((823,y+62), f"Facing {start['yaw_degrees']:+.0f}°", font=font(19), fill='#e3eaf3')
                d.text((45, 548), 'Initial pose held' if frame < 60 else 'Recorded motion replay from the chosen mark', font=font(19), fill='#f2f7ff')
                d.text((36, 615), 'Live Neon plan + validated staging + saved motion', font=font(22), fill='#e2edf7')
                d.text((36, 655), 'Placement preview only. No fresh Core / InterGen generation or contact-quality validation.', font=font(18), fill='#a9bbce')
                if frame == 0:
                    thumbnails.append(im.copy())
                encoder.stdin.write(im.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')
    finally:
        if encoder.poll() is None:
            encoder.kill(); encoder.wait()
    sheet = Image.new('RGB', (1280, 720*len(thumbnails)))
    for i, im in enumerate(thumbnails):
        sheet.paste(im, (0, i*720))
    sheet.save(destination.with_suffix('.jpg'), quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway-env', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-plans', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'plans.json'
    if args.replay_plans:
        records = json.loads(target.read_text())['cases']
    else:
        if args.gateway_env:
            for line in args.gateway_env.read_text().splitlines():
                parts = shlex.split(line, comments=True)
                if len(parts) == 1 and '=' in parts[0]:
                    key, value = parts[0].split('=', 1)
                    if key in ('NEON_AI_GATEWAY_BASE_URL', 'NEON_AI_GATEWAY_TOKEN'):
                        os.environ[key] = value
        gateway = GatewayGenerator.from_env(stage='layout'); gateway.model = 'gpt-6-astra'
        planner = ScenePromptPlanner(gateway)
        records = []
        for title, prompt in CASES:
            plan = planner.plan(prompt, SCENE)
            records.append({'title': title, 'plan': plan, 'placement': auto_place(plan, SCENE)})
            print(title, flush=True)
        target.write_text(json.dumps({'scene': SCENE, 'cases': records}, indent=2)+'\n')
    source_path = ROOT/'review/prompt-scenes/solo-city-capture/scene.cast.stagezero.npz'
    with np.load(source_path, allow_pickle=False) as archive:
        source = archive['joints'][:, 0]
    render(records, source, args.output/'placement-review.mp4')
    (args.output/'provenance.json').write_text(json.dumps({'kind':'placement preview, recorded motion replay',
        'live_planner_model':'gpt-6-astra', 'fresh_motion_generation':False,
        'source':str(source_path.relative_to(ROOT)), 'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'cases':len(records), 'seconds':6*len(records), 'fps':30}, indent=2)+'\n')


if __name__ == '__main__':
    main()
