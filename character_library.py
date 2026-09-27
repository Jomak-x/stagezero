"""Local, reusable character assets, separate from motion project files."""
import io
import json
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
import trimesh
from PIL import Image

from character_generation import validate_glb


def preview_transform(data, height=1.7):
    """Fit a Y-up GLB beside the actor without modifying its materials or rig."""
    validate_glb(data)
    try:
        scene = trimesh.load(io.BytesIO(data), file_type='glb', force='scene', process=False)
        bounds = np.asarray(scene.bounds, dtype=float)
    except Exception:
        raise ValueError('Cannot read character geometry; use a self-contained GLB with mesh geometry') from None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError('Character must contain finite mesh geometry')
    extent = bounds[1] - bounds[0]
    if extent[1] < 1e-6 or max(extent) / extent[1] > 20:
        raise ValueError('Character must have a nonzero height and upright Y-up proportions')
    scale = height / extent[1]
    center = (bounds[0] + bounds[1]) / 2
    return float(scale), (float(2.5 - center[0] * scale), float(-bounds[0, 1] * scale), float(-center[2] * scale))


class CharacterLibrary:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, identifier, suffix):
        if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('Invalid character identifier')
        path = self.folder / (identifier + suffix)
        if path.is_symlink():
            raise ValueError('Character files must be local library files')
        return path

    def save(self, data, prompt, source, reference=None):
        preview_transform(data)
        if reference is not None:
            self._validate_reference(reference)
        identifier = uuid.uuid4().hex
        metadata = {'id': identifier, 'name': ' '.join(str(prompt).split())[:70] or 'Imported character',
                    'prompt': str(prompt)[:800], 'source': source, 'animation_ready': False}
        model = self._path(identifier, '.glb')
        manifest = self._path(identifier, '.json')
        reference_path = self._path(identifier, '.png')
        try:
            model.write_bytes(data)
            if reference is not None:
                reference_path.write_bytes(reference)
            manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        except OSError:
            model.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
            reference_path.unlink(missing_ok=True)
            raise
        return metadata

    def entries(self):
        entries = []
        for path in sorted(self.folder.glob('*.json')):
            try:
                if path.is_symlink() or path.stat().st_size > 8192:
                    continue
                doc = json.loads(path.read_text())
                if not isinstance(doc, dict) or doc.get('id') != path.stem or not isinstance(doc.get('name'), str):
                    continue
                if self._path(doc['id'], '.glb').is_file():
                    entries.append(doc)
            except (OSError, ValueError):
                continue
        return entries

    def read(self, identifier):
        path = self._path(identifier, '.glb')
        with path.open('rb') as stream:
            data = stream.read(40 * 1024 * 1024 + 1)
        validate_glb(data)
        return data

    def set_active(self, identifier):
        if identifier is not None:
            self._path(identifier, '.glb')
        path = self.folder / '.active.json'
        staged = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.folder, prefix='.active-',
                                             suffix='.tmp', delete=False) as stream:
                staged = Path(stream.name)
                stream.write(json.dumps({'id': identifier}) + '\n')
            staged.replace(path)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)

    def active(self):
        try:
            path = self.folder / '.active.json'
            if path.is_symlink() or path.stat().st_size > 1024:
                return None
            identifier = json.loads(path.read_text()).get('id')
            if identifier and self._path(identifier, '.glb').is_file():
                return identifier
        except (OSError, ValueError, AttributeError):
            pass
        return None

    @staticmethod
    def _validate_reference(data):
        if not isinstance(data, bytes) or len(data) > 10 * 1024 * 1024:
            raise ValueError('Character reference image is too large')
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'PNG' or image.width * image.height > 16_000_000:
                    raise ValueError('Character reference must be a bounded PNG')
                image.verify()
        except (OSError, SyntaxError):
            raise ValueError('Character reference PNG is invalid') from None

    def read_reference(self, identifier):
        path = self._path(identifier, '.png')
        if not path.exists():
            return None
        with path.open('rb') as stream:
            data = stream.read(10 * 1024 * 1024 + 1)
        self._validate_reference(data)
        return data
