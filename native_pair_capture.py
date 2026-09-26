"""Full-frame studio playback export; model generation is never invoked."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw
from experiments.capture_core_performance import get_render_with_timeout


def capture_pair(session, renderer, client, output_dir, *, flush=None, render_lock=None, render_frame=None):
    if client is None:
        raise ValueError('Connect a browser to export playback.')
    state = session.snapshot()
    if not state['active'] or state['busy'] or not state['total_frames']:
        raise ValueError('Open a complete interaction and finish generation before exporting.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Acquisition shares the viewer's render transaction lock, so a live scene
    # update cannot pass its capture check immediately before this lease begins.
    with render_lock or nullcontext():
        state = session.begin_capture()
    try:
        return _capture_locked(session, renderer, client, output, state, flush, render_lock, render_frame)
    finally:
        with render_lock or nullcontext():
            session.end_capture(state['frame'])


def _capture_locked(session, renderer, client, output, state, flush, render_lock, render_frame):
    clip = session.timeline_clip()
    with render_lock or nullcontext():
        renderer.sync_cast(state)
        renderer.set_clip(clip)
    archive = session.save()
    (output / 'scene.native-pair.stagezero.npz').write_bytes(archive)
    frames, fps = state['total_frames'], state['fps']
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output / 'playback.mp4')], stdin=subprocess.PIPE)
    samples = []
    indices = set(np.linspace(0, frames-1, min(12,frames)).round().astype(int).tolist())
    completed = 0
    try:
        for frame in range(frames):
            current = session.snapshot()
            if session.timeline_clip() is not clip or not current['active'] or current['busy'] or current['revision'] != state['revision']:
                raise RuntimeError('Interaction changed during export; partial capture retained.')
            with render_lock or nullcontext():
                session.capture_seek(frame)
                renderer.tick(frame)
                if render_frame is not None:
                    render_frame(frame, state)
                if flush is not None:
                    flush()
                pixels = get_render_with_timeout(client, width=1280, height=720, timeout=30.)
            pixels = np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
            if pixels.shape != (720, 1280, 3):
                raise ValueError('Browser returned an invalid capture size.')
            encoder.stdin.write(pixels.tobytes())
            if frame in (0, frames-1):
                Image.fromarray(pixels).save(output / ('first-frame.png' if frame == 0 else 'last-frame.png'))
            completed += 1
            if frame in indices:
                samples.append((frame, Image.fromarray(pixels).resize((400,225))))
        encoder.stdin.close()
        if encoder.wait(timeout=30) != 0:
            raise RuntimeError('Video encoder failed.')
        sheet = Image.new('RGB', (1600, 253*((len(samples)+3)//4)), (18,24,31))
        draw = ImageDraw.Draw(sheet)
        for n,(frame,picture) in enumerate(samples):
            x,y=(n%4)*400,(n//4)*253
            sheet.paste(picture,(x,y+28)); draw.text((x+8,y+7),f'{frame/fps:.2f}s · frame {frame}',fill='white')
        sheet.save(output/'contact-sheet.png')
        metadata = {'frames':frames,'fps':fps,'duration_seconds':frames/fps,'frame_indices':list(range(frames)),
            'capture':'Complete studio playback, one render per source frame; no cuts or time edits',
            'archive_sha256':hashlib.sha256(archive).hexdigest(),'cast':state.get('cast'),
            'selected_pair':state.get('selected_pair'),'captured_at':time.time()}
        (output/'capture-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return output/'playback.mp4'
    except Exception as exc:
        (output/'capture-failure.json').write_text(json.dumps({'completed_frames':completed,'error':str(exc)}))
        raise
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
