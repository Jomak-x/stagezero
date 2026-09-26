"""Small, reproducible gallery images rendered from the real preset geometry.

No network calls or invented reference pictures: these use the same mesh recipes
as the live viewport. Lighting/effects are simplified for a readable thumbnail.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
from PIL import Image, ImageDraw
from asset_geometry import compile_asset
from scene_composition import PRESETS, make_preset

PREVIEW_DIRECTORY = Path(__file__).resolve().parent / 'assets' / 'background-previews'


def preview_path(name):
    return PREVIEW_DIRECTORY / (name.lower().replace(' ', '-') + '.jpg')


@lru_cache(maxsize=16)
def preset_preview(name):
    path = preview_path(name)
    if path.exists():
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB')).copy()
    return render_preset(name)


class _Meshes:
    def add_box(self, name, dimensions, color, **kwargs):
        import trimesh
        mesh = trimesh.creation.box(extents=dimensions)
        return self.add_mesh_simple(name, mesh.vertices, mesh.faces, color)

    def add_icosphere(self, name, radius, color, subdivisions=1, **kwargs):
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        return self.add_mesh_simple(name, mesh.vertices, mesh.faces, color)

    def add_mesh_simple(self, name, vertices, faces, color, **kwargs):
        return SimpleNamespace(vertices=np.asarray(vertices), faces=np.asarray(faces), color=color, wxyz=None)


def render_preset(name, width=560, height=300):
    """Flat-shaded orthographic preview, using actual per-part preset colors."""
    from object_scene import ObjectSceneLayer
    document = make_preset(name)
    assets = {a['id']: a for a in document.get('assets', [])}
    layer = ObjectSceneLayer(SimpleNamespace(scene=_Meshes()))
    triangles, colors = [], []
    compiled = {}
    for obj in document['objects']:
        yaw = math.radians(obj.get('yaw', 0))
        rotation = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        if obj['kind'] == 'custom':
            if obj['asset'] not in compiled:
                compiled[obj['asset']] = compile_asset(assets[obj['asset']])
            vertices, faces, rgb = compiled[obj['asset']]
            lower, upper = vertices.min(0), vertices.max(0)
            vertices = (vertices - (lower + upper) / 2) / np.maximum(upper - lower, 1e-6) * obj['size']
            triangles.append((vertices @ rotation.T + obj['position'])[faces])
            colors.append(rgb[faces].mean(1))
        else:
            for handle, offset, _ in layer._build(obj):
                # _build already rotates each part's offset by object yaw.
                vertices = handle.vertices @ rotation.T + np.asarray(obj['position']) + offset
                triangles.append(vertices[handle.faces])
                colors.append(np.tile(handle.color, (len(handle.faces), 1)))
    triangles = np.concatenate(triangles)
    colors = np.concatenate(colors)
    # A consistent three-quarter overview keeps every preset directly comparable.
    forward = np.array([.7, .62, 1.2]); forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    projected = triangles @ np.stack([right, up, forward], axis=1)
    lo, hi = projected[:, :, :2].min((0, 1)), projected[:, :, :2].max((0, 1))
    scale = min((width - 42) / (hi[0] - lo[0]), (height - 36) / (hi[1] - lo[1]))
    xy = (projected[:, :, :2] - (lo + hi) / 2) * scale
    xy[:, :, 0] += width / 2
    xy[:, :, 1] = height / 2 - xy[:, :, 1]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    light = np.array([-.5, 1., .7]); light /= np.linalg.norm(light)
    shade = .57 + .43 * np.maximum(0, normals @ light)
    colors = np.clip(colors * shade[:, None], 0, 255).astype(np.uint8)
    # Render at twice the size to retain thin rails and architectural detail.
    canvas = Image.new('RGB', (width * 2, height * 2), '#172832')
    draw = ImageDraw.Draw(canvas)
    for index in np.argsort(projected[:, :, 2].mean(1)):
        draw.polygon([tuple(v) for v in xy[index] * 2], fill=tuple(colors[index]))
    return np.asarray(canvas.resize((width, height), Image.Resampling.LANCZOS))


if __name__ == '__main__':
    PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS:
        Image.fromarray(render_preset(preset)).save(preview_path(preset), quality=88)
        print(preset, flush=True)
