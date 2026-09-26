"""Procedural Viser models for validated scene objects.

Every part stays inside the object's declared, axis-aligned size. Static
details keep their own colors when an interaction changes the main material.
"""

import json
import math
import hashlib
import inspect

import numpy as np

from asset_geometry import compile_asset, mesh_to_glb, validate_assets


def _shade(color, factor):
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


class ObjectSceneLayer:
    def __init__(self, server):
        self.server = server
        self.signature = None
        self.handles = {}
        self._atmosphere = None
        self._assets = {}
        self._asset_digests = {}
        self._object_signatures = {}
        self._object_snapshot = None
        self._asset_snapshot = None
        self._state_snapshot = {}
        self._mesh_cache = {}
        self._glb_cache = {}
        self._glb_shadow_argument = None
        self._target_layer = None

    def update(self, objects, states):
        """Draw a state list or an object/effect/lighting playback bundle.

        The bundle is {'objects': states, 'effects': specs,
        'seconds': elapsed, 'lighting': preset}. A legacy list clears effects.
        """
        bundle = isinstance(states, dict)
        targets=states.get('targets',[]) if bundle else []
        if targets or self._target_layer is not None:
            if self._target_layer is None:
                from scene_targets import TargetLayer
                self._target_layer=TargetLayer(self.server)
            self._target_layer.update(targets,objects)
        if bundle:
            effect_specs = states.get('effects', [])
            seconds = states.get('seconds', 0.0)
            lighting = states.get('lighting', 'neutral')
            raw_assets = states.get('assets', [])
            states = states['objects']
        else:
            effect_specs, seconds, lighting = [], 0.0, 'neutral'
            raw_assets = []
        if self._atmosphere is None and bundle:
            from scene_atmosphere import SceneAtmosphereLayer
            self._atmosphere = SceneAtmosphereLayer(self.server)
        if self._atmosphere is not None:
            self._atmosphere.update(effect_specs, seconds, lighting)

        # Keep detached snapshots so edits made in place are still detected.
        # Equality is much cheaper than serializing every asset each frame.
        if objects != self._object_snapshot or raw_assets != self._asset_snapshot:
            assets = validate_assets(raw_assets)
            available = {asset['id'] for asset in assets}
            for obj in objects:
                if obj['kind'] == 'custom' and obj['asset'] not in available:
                    raise ValueError(f"custom object {obj['id']} references missing asset {obj['asset']}")
            asset_digests = {
                asset['id']: hashlib.sha256(json.dumps(asset, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                for asset in assets
            }
            signatures = {}
            for obj in objects:
                # Position and interaction affect state, not mesh geometry.
                shape = (obj['kind'], tuple(obj['size']), obj.get('yaw', 0))
                if obj['kind'] == 'custom':
                    shape += (obj['asset'], asset_digests[obj['asset']])
                else:
                    shape += (tuple(obj['color']),)
                signatures[obj['id']] = shape
            changed = {identifier for identifier, shape in signatures.items()
                       if self._object_signatures.get(identifier) != shape}
            for identifier in set(self.handles) - set(signatures) | changed:
                for handle, _, _ in self.handles.pop(identifier, []):
                    handle.remove()
                self._state_snapshot.pop(identifier, None)
            self._assets = {asset['id']: asset for asset in assets}
            self._asset_digests = asset_digests
            for obj in objects:
                if obj['id'] in changed:
                    self.handles[obj['id']] = self._build(obj)
            self._object_signatures = signatures
            self._object_snapshot = json.loads(json.dumps(objects))
            self._asset_snapshot = json.loads(json.dumps(raw_assets))
            self.signature = (self._object_snapshot, self._asset_snapshot)
        for state in states:
            current = (tuple(state['position']), tuple(state['color']), state['active'])
            if self._state_snapshot.get(state['id']) == current:
                continue
            for handle, offset, role in self.handles.get(state['id'], []):
                handle.position = tuple(a + b for a, b in zip(state['position'], offset))
                if role == 'body':
                    handle.color = tuple(state['color'])
                elif role == 'screen':
                    handle.color = (87, 242, 222) if state['active'] else (34, 88, 110)
            self._state_snapshot[state['id']] = current

    def _build(self, obj):
        if obj['kind'] == 'custom':
            return self._build_custom(obj)
        x, y, z = obj['size']
        name = '/objects/' + obj['id']
        main = tuple(obj['color'])
        dark = _shade(main, .61)
        light = _shade(main, 1.25)
        parts = []

        def box(suffix, dimensions, offset=(0, 0, 0), color=None, role='detail'):
            handle = self.server.scene.add_box(name + '/' + suffix,
                                               dimensions=dimensions, color=color or main)
            parts.append((handle, offset, role))

        def sphere(suffix, radius, offset=(0, 0, 0), color=None, role='detail', subdivisions=1):
            handle = self.server.scene.add_icosphere(name + '/' + suffix, radius=radius,
                                                     color=color or main, subdivisions=subdivisions,
                                                     flat_shading=subdivisions == 1)
            parts.append((handle, offset, role))

        def cylinder(suffix, radius, height, offset=(0, 0, 0), color=None, role='detail'):
            vertices = []
            for level in (-height / 2, height / 2):
                vertices.extend((radius * math.cos(2 * math.pi * i / 12), level,
                                 radius * math.sin(2 * math.pi * i / 12)) for i in range(12))
            faces = []
            for i in range(12):
                j = (i + 1) % 12
                faces.extend(((i, j, 12 + i), (j, 12 + j, 12 + i)))
            faces.extend((0, i + 1, i) for i in range(1, 11))
            faces.extend((12, 12 + i, 12 + i + 1) for i in range(1, 11))
            handle = self.server.scene.add_mesh_simple(name + '/' + suffix,
                                                        vertices=np.asarray(vertices, dtype=np.float32),
                                                        faces=np.asarray(faces, dtype=np.uint32),
                                                        color=color or main, flat_shading=True)
            parts.append((handle, offset, role))

        def legs(suffix, height, thickness=.1, spread_x=.4, spread_z=.4):
            for i, sx in enumerate((-1, 1)):
                for j, sz in enumerate((-1, 1)):
                    box(f'{suffix}{i}{j}', (x * thickness, height, z * thickness),
                        (sx * x * spread_x, -y / 2 + height / 2, sz * z * spread_z), dark)

        kind = obj['kind']
        if kind == 'ball':
            sphere('ball', x / 2, role='body', subdivisions=3)
        elif kind == 'door':
            box('panel', (x * .84, y * .92, z * .55), role='body')
            box('header', (x, y * .04, z), (0, y * .48, 0), dark)
            for i, sx in enumerate((-1, 1)):
                box(f'jamb{i}', (x * .08, y, z), (sx * x * .46, 0, 0), dark)
            sphere('handle', min(x, y, z) * .16, (x * .28, -y * .04, z * .32),
                   (213, 182, 107))
        elif kind == 'chair':
            box('seat', (x * .88, y * .11, z * .82), (0, -y * .03, -z * .02), role='body')
            box('back', (x * .88, y * .48, z * .11), (0, y * .25, z * .36), role='body')
            legs('leg', y * .43, .1, .38, .37)
        elif kind == 'lamp':
            cylinder('base', min(x, z) * .48, y * .07, (0, -y * .465, 0), dark)
            cylinder('stem', min(x, z) * .075, y * .69, (0, -y * .09, 0), dark)
            cylinder('shade', min(x, z) * .48, y * .27, (0, y * .36, 0), light, 'body')
            sphere('bulb', min(x, y, z) * .12, (0, y * .32, 0), (255, 229, 157))
        elif kind == 'table':
            box('top', (x, y * .10, z), (0, y * .45, 0), role='body')
            legs('leg', y * .90, .08, .43, .42)
            box('apron', (x * .78, y * .10, z * .06), (0, y * .34, -z * .4), dark)
        elif kind == 'sofa':
            box('base', (x, y * .25, z * .83), (0, -y * .20, -z * .05), dark)
            box('back', (x, y * .70, z * .20), (0, y * .15, z * .40), role='body')
            for i, sx in enumerate((-1, 1)):
                box(f'arm{i}', (x * .15, y * .45, z * .78),
                    (sx * x * .425, -y * .05, -z * .08), role='body')
            for i, dx in enumerate((-.22, .22)):
                box(f'cushion{i}', (x * .42, y * .19, z * .60),
                    (dx * x, -y * .03, -z * .09), light)
        elif kind == 'crate':
            box('body', (x * .96, y * .96, z * .96), role='body')
            for i, level in enumerate((-.40, .40)):
                box(f'front_rail{i}', (x, y * .09, z * .04), (0, level * y, -z * .48), dark)
                box(f'back_rail{i}', (x, y * .09, z * .04), (0, level * y, z * .48), dark)
            for i, sx in enumerate((-1, 1)):
                box(f'front_post{i}', (x * .09, y, z * .04), (sx * x * .45, 0, -z * .48), dark)
        elif kind == 'barrel':
            cylinder('staves', min(x, z) * .46, y * .96, role='body')
            for i, level in enumerate((-.35, .35)):
                cylinder(f'band{i}', min(x, z) * .50, y * .09, (0, level * y, 0),
                         (74, 81, 88))
            cylinder('lid', min(x, z) * .46, y * .025, (0, y * .48, 0), dark)
        elif kind == 'pillar':
            box('shaft', (x * .65, y * .83, z * .65), role='body')
            box('base', (x, y * .09, z), (0, -y * .455, 0), dark)
            box('capital', (x, y * .08, z), (0, y * .46, 0), light)
        elif kind == 'wall':
            box('surface', (x, y * .95, z * .84), role='body')
            box('cap', (x, y * .05, z), (0, y * .475, 0), light)
            box('foot', (x, y * .07, z), (0, -y * .465, 0), dark)
            for i in range(1, 5):
                box(f'seam{i}', (x * .005, y * .83, z * .02),
                    (x * (i / 5 - .5), 0, -z * .43), dark)
        elif kind == 'arch':
            for i, sx in enumerate((-1, 1)):
                box(f'post{i}', (x * .19, y * .81, z * .90),
                    (sx * x * .405, -y * .095, 0), main, 'body')
                box(f'base{i}', (x * .23, y * .06, z),
                    (sx * x * .385, -y * .47, 0), dark)
            box('lintel', (x, y * .19, z), (0, y * .405, 0), role='body')
            box('keystone', (x * .09, y * .21, z), (0, y * .39, 0), light)
        elif kind == 'plant':
            cylinder('pot', min(x, z) * .31, y * .37, (0, -y * .315, 0), (137, 83, 61))
            cylinder('rim', min(x, z) * .36, y * .06, (0, -y * .16, 0), (163, 106, 75))
            cylinder('stem', min(x, z) * .05, y * .46, (0, y * .09, 0), dark)
            for i, (dx, dz) in enumerate(((-.22, 0), (.22, 0), (0, -.22), (0, .22))):
                sphere(f'leaf{i}', min(min(x, z) * .21, y * .15),
                       (x * dx, y * (.26 if i % 2 else .35), z * dz),
                       light if i % 2 else main)
        elif kind == 'tree':
            cylinder('trunk', min(x, z) * .12, y * .60, (0, -y * .2, 0), (104, 75, 52))
            for i, (dx, dz, level, factor) in enumerate((
                (0, 0, .30, 1.0), (-.22, 0, .16, .9), (.22, 0, .16, .85),
                (0, -.22, .18, .82), (0, .22, .18, .9))):
                sphere(f'crown{i}', min(min(x, z) * .27, y * .20),
                       (x * dx, y * level, z * dz), _shade(main, factor))
        elif kind == 'rock':
            vertices = np.asarray([(0, y * .5, 0), (-x * .5, 0, 0),
                                   (0, 0, -z * .5), (x * .5, 0, 0),
                                   (0, 0, z * .5), (0, -y * .5, 0)], dtype=np.float32)
            faces = np.asarray([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
                                (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)], dtype=np.uint32)
            handle = self.server.scene.add_mesh_simple(name + '/rock', vertices=vertices,
                                                        faces=faces, color=main, flat_shading=True)
            parts.append((handle, (0, 0, 0), 'body'))
        elif kind == 'console':
            box('pedestal', (x * .68, y * .72, z * .70), (0, -y * .14, 0), role='body')
            box('face', (x, y * .34, z * .92), (0, y * .32, 0), dark)
            box('screen', (x * .70, y * .22, z * .02),
                (0, y * .34, -z * .47), (34, 88, 110), 'screen')
            for i, dx in enumerate((-.33, -.20, -.07)):
                box(f'key{i}', (x * .06, y * .035, z * .035),
                    (x * dx, y * .115, -z * .45), (134, 165, 174))
        elif kind == 'platform':
            box('deck', (x, y * .72, z), (0, y * .14, 0), role='body')
            box('trim_front', (x, y * .16, z * .03), (0, -y * .42, -z * .485), light)
            box('trim_back', (x, y * .16, z * .03), (0, -y * .42, z * .485), light)
        else:
            raise ValueError(f'unsupported object kind: {kind}')
        if obj.get('yaw', 0):
            angle = math.radians(obj['yaw'])
            c, s = math.cos(angle), math.sin(angle)
            rotated = []
            for handle, offset, role in parts:
                handle.wxyz = (math.cos(angle / 2), 0., math.sin(angle / 2), 0.)
                dx, dy, dz = offset
                rotated.append((handle, (c*dx+s*dz, dy, -s*dx+c*dz), role))
            parts = rotated
        return parts

    def _build_custom(self, obj):
        """Create one colored GLB handle from a validated asset recipe."""
        asset = self._assets.get(obj['asset'])
        if asset is None:
            raise ValueError(f"custom object {obj['id']} references missing asset {obj['asset']}")
        digest = self._asset_digests[obj['asset']]
        if digest not in self._mesh_cache:
            self._mesh_cache[digest] = compile_asset(asset)
            if len(self._mesh_cache) > 32:
                oldest = next(iter(self._mesh_cache))
                if oldest != digest:
                    del self._mesh_cache[oldest]
        size = tuple(obj['size'])
        glb_key = (digest, size)
        if glb_key not in self._glb_cache:
            vertices, faces, colors = self._mesh_cache[digest]
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            normalized = (vertices - (lower + upper) / 2) / np.maximum(upper - lower, 1e-6)
            self._glb_cache[glb_key] = mesh_to_glb(normalized * np.asarray(size, dtype=np.float32), faces, colors)
            if len(self._glb_cache) > 64:
                oldest = next(iter(self._glb_cache))
                if oldest != glb_key:
                    del self._glb_cache[oldest]
        yaw = math.radians(obj.get('yaw', 0)) / 2
        kwargs = {'wxyz': (math.cos(yaw), 0., math.sin(yaw), 0.)}
        # Large compound assets have many intersecting box faces. Viser's
        # shadow receiver produces visible self-shadow stripes on their broad
        # horizontal or vertical surfaces; they still cast shadows on the
        # floor and smaller props.
        if self._glb_shadow_argument is None:
            parameters = inspect.signature(self.server.scene.add_glb).parameters
            self._glb_shadow_argument = ('receive_shadow' in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()))
        if self._glb_shadow_argument:
            broadest_face = max(size[0] * size[1], size[0] * size[2], size[1] * size[2])
            kwargs['receive_shadow'] = broadest_face < 4.0
        handle = self.server.scene.add_glb('/objects/' + obj['id'], self._glb_cache[glb_key], **kwargs)
        return [(handle, (0, 0, 0), 'custom')]
