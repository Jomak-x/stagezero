"""Two-stage scene production: invent reusable geometry, then compose a set."""
import copy
import json
import numpy as np
from asset_geometry import _rotation
from asset_geometry import validate_assets
from scene_asset_library import AssetLibrary
from scene_objects import KINDS, make_object
from scene_generation import validate_generated_scene
from scene_effects import EFFECT_KINDS, make_effect
from object_generation import GatewayGenerator, validate_prompt


def fit_generated_assets(assets):
    """Fit generated assemblies to their unit box; imported scenes stay strict.

    Validate all fields first, then normalize only
    assets whose detail geometry overshoots the requested local bounds.
    """
    try:
        return validate_assets(assets)
    except ValueError as exc:
        if 'extends beyond the local unit bounds' not in str(exc):
            raise
    validate_assets(assets, check_bounds=False)
    result = copy.deepcopy(assets)
    for asset in result:
        lows, highs = [], []
        for part in asset['parts']:
            extent = np.abs(_rotation(part.get('rotation', [0,0,0]))) @ (np.array(part['size']) / 2)
            first = np.array(part['position'])
            repeat = part.get('repeat', {'count':[1,1,1], 'step':[0,0,0]})
            last = first + (np.array(repeat['count'])-1)*np.array(repeat['step'])
            lows.append(np.minimum(first,last)-extent)
            highs.append(np.maximum(first,last)+extent)
        low, high = np.min(lows,axis=0), np.max(highs,axis=0)
        if np.all(low >= -.5) and np.all(high <= .5): continue
        center = (low+high)/2
        scale = min(1., .999 / float(np.max(high-low)))
        for part in asset['parts']:
            part['position'] = ((np.array(part['position'])-center)*scale).tolist()
            part['size'] = (np.array(part['size'])*scale).tolist()
            if 'repeat' in part:
                part['repeat']['step'] = (np.array(part['repeat']['step'])*scale).tolist()
    return validate_assets(result)


def fit_generated_layout(doc):
    """Give thin floor/rug platforms renderable thickness, keeping their tops."""
    doc = copy.deepcopy(doc)
    if isinstance(doc, dict) and isinstance(doc.get('objects'), list):
        for obj in doc['objects']:
            if not isinstance(obj, dict) or obj.get('kind') != 'platform': continue
            size, position = obj.get('size'), obj.get('position')
            if (isinstance(size, list) and len(size) == 3 and isinstance(position, list)
                    and len(position) == 3 and type(size[1]) in (int,float)
                    and type(position[1]) in (int,float) and .005 <= size[1] < .05):
                position[1] -= (.05-size[1])/2
                size[1] = .05
    return doc


def asset_system_prompt():
    return '''You are a stylized 3D prop designer. Create 6 to 8 reusable original assets tailored to the user's scene. A cohesive set should include several distinct anchor structures and smaller props that establish scale and activity.
Return JSON only: {"assets":[{"id":"safe-id","name":"Descriptive name","parts":[...]}]}.
Each part has EXACTLY: shape (box,sphere,cylinder,cone), position [x,y,z], size [width,height,depth], color [R,G,B] integer 0..255.
Optional rotation [rx,ry,rz] in degrees. Optional repeat {"count":[nx,ny,nz],"step":[dx,dy,dz]} adds a lattice starting at position.
Use normalized LOCAL coordinates: every vertex including repeated/rotated shapes must fit in [-0.5,0.5] on EACH axis.
Y is UP. Cylinder/cone axis is Y. Objects face +Z; place facade/front details toward +Z.
The scene composer later scales each entire asset to meters, so all parts must use local proportions, never world meters.
Use up to 32 parts per asset, target fewer than 180 total expanded shapes per asset (512 is an absolute limit), and 8..22 parts for visually distinctive important assets. A scene may reuse a building many times, so prefer carefully spaced 3x4 window grids to dense 8x12 grids. Spend the most detail on the anchor structures; vary silhouettes and color accents so repeated instances do not look identical.
Use detail levels: main silhouette, secondary structures, small accents. Do NOT output one featureless cube per asset.
Use repeated windows for buildings, frames for storefronts, beams and roof details, upholstery for furniture, shelves and books.
For a repeated window grid, the window width and height MUST each be smaller than their repeat step by at least 0.02 local units, leaving visible facade strips. Example: size [0.09,0.08,0.012], repeat step [0.16,0.14,0]. Never make a continuous glass slab when the request calls for individual windows.
The frontmost visible face of a facade is +Z. Place windows and doors just IN FRONT of the wall surface: window frontZ must exceed the facade frontZ by at least 0.003 local units. Do not put a solid wall panel, dark slab, or trim box in front of the windows. A thin inset panel is acceptable only when its visible face remains in front of the facade.
Keep a harmonious palette and sensible proportions. Use midtone facade and furniture colors (RGB channels roughly 90..210), reserve very dark colors for small trim. Connect roofs and upper stories to the main structure: no floating ornament. Windows can be inset colored panels or thin boxes with surrounding trim.
Reserve detail for what defines the prop. Assets can be ANY object the scene needs, not a predefined catalog.
No scripts, URLs, textures, shaders or external files. No extra keys. Never generate a whole city as one asset: create several building types and street props to instance.
Example part grid: {"shape":"box","position":[-0.3,-0.2,0.455],"size":[0.12,0.09,0.015],"color":[95,163,201],"repeat":{"count":[4,5,1],"step":[0.2,0.13,0]}}.
For a room, create INDIVIDUAL furniture/storage/lights that match the request: a bedroom needs a bed and bedside furniture, a kitchen needs counters and appliances, and a workshop needs benches and tools. Never combine two furniture items into one asset, and never generate the entire room or its walls as one asset. Architecture is supplied by the composer. For an outdoor city or neighborhood, prioritize at least three distinct detailed buildings, a storefront or landmark, and street furniture. A market should include multiple different stalls, canopies, display goods and a focal point such as a fountain. An industrial space should include separate large equipment, workstations and storage.
'''


def layout_system_prompt(assets):
    summaries = [{'id': a['id'], 'name': a['name']} for a in assets]
    catalog = [make_object(k, i) for i,k in enumerate(KINDS)]
    return ('''You are an environment artist composing a coherent 3D scene for a 1.3m humanoid at world origin.
Return JSON only: {"version":3,"name":"Title","assets":[],"objects":[...],"effects":[],"lighting":"neutral","camera":{"position":[8,4,12],"look_at":[0,1,-3]}}.
The application supplies the asset definitions: keep assets=[] in your response; reference their exact IDs in objects.
Custom object EXACT fields: {"id":"unique-safe-id","name":"Short readable name","kind":"custom","asset":"provided-asset-id","position":[x,y,z],"size":[width,height,depth],"color":[255,255,255],"interaction":{"action":"none","trigger":"none","radius":0},"yaw":0}.
Yaw is degrees around Y. Position is CENTER in meters, Y up. Size scales normalized geometry to world dimensions. Ground rests at y=0 (centerY=height/2).
Use 8-25 total instances as appropriate; do not duplicate whole rooms or large furniture to meet a count. Reuse architectural props at several sizes and locations. You may also use catalog kinds for floors,walls or functional props (same fields but omit asset; optional yaw is supported).
Center actor area x=-1.2..1.2,z=-1.2..1.2 must stay clear. Camera is +Z. Tall scenery belongs behind actor at negative Z and to sides. Do NOT block view with foreground furniture.
City scenes need wide roads, raised sidewalks, believable near buildings and smaller/distant skyline layers, alleys, storefront rhythm and street furniture. Never arrange buildings in a random circle.
Room scenes need floor, rear/side walls with open front for viewing, architectural detail, coherent furniture group, rug and small objects. Do NOT build a front wall.
Floor top should be y=.02 to cover the old demo platform. Camera should keep actor readable while showing background, not zoom out for entire skyline.
Coordinates +/-100m; dimensions .05..60m. Keep near-city buildings 4-12m high and distant ones 10-25m. Furniture height .5-2m. Avoid intersecting solids except purposeful attachments.
Lighting one of neutral,warm,moonlight,neon,sunset. Prefer warm for interiors, neutral or sunset for cities. effects=[] is fine: good architecture matters more than particles.
The whole scene must stay below 250000 triangles (box=12, sphere=168, cylinder=48, cone=24 per repeated shape), 12000 shape instances and 64 objects. No arbitrary asset IDs, code, URLs, extra fields or objects unsupported by the prepared assets/catalog. Use meaningful composition, scale variation, matched colors.
PREPARED ASSETS: ''' + json.dumps(summaries) + '\nCATALOG: ' + json.dumps(catalog) + '\nEFFECT CATALOG: ' + json.dumps([make_effect(k, i) for i, k in enumerate(EFFECT_KINDS)]) + '\nOnly add effects when requested or appropriate to the scene. Explosion and energy_burst are six-second cinematic loops. Effect positions are centers, within +/-20m, sizes .1..8m, intensity 0..1, max 8 effects. Keep the actor visible.')


class AdaptiveSceneGenerator:
    def __init__(self, gateway=None, library=None, prepared_assets=None, progress=None):
        self.gateway = gateway if gateway is not None else GatewayGenerator.from_env(stage='assets')
        self.layout_gateway = gateway if gateway is not None else GatewayGenerator.from_env(stage='layout')
        self.library = library if library is not None else AssetLibrary()
        self.prepared_assets = self._review_assets(validate_assets(prepared_assets)) if prepared_assets is not None else None
        self.progress = progress or (lambda message: None)

    @staticmethod
    def _review_assets(assets):
        from asset_quality import assess_assets, refine_assets
        reviewed = refine_assets(assets)
        issues = [issue for issue in assess_assets(reviewed) if issue['severity'] == 'error']
        if any(issue['code'] == 'facade_occlusion' for issue in issues):
            # A pane at the unit-box edge needs space to move ahead of its
            # backing. The renderer fits actual bounds, so reserving this
            # margin preserves world proportions rather than shrinking props.
            affected = {issue['asset_id'] for issue in issues if issue['code'] == 'facade_occlusion'}
            for asset in reviewed:
                if asset['id'] not in affected:
                    continue
                for part in asset['parts']:
                    part['position'] = [v * .9 for v in part['position']]
                    part['size'] = [max(.001, v * .9) for v in part['size']]
                    if 'repeat' in part:
                        part['repeat']['step'] = [v * .9 for v in part['repeat']['step']]
            reviewed = refine_assets(reviewed)
            issues = [issue for issue in assess_assets(reviewed) if issue['severity'] == 'error']
        if issues:
            summary = '; '.join(f"{issue['asset_id']}: {issue['message']}" for issue in issues[:6])
            raise ValueError('Generated prop quality check failed: ' + summary)
        return reviewed

    @staticmethod
    def _model_label(gateway):
        model = getattr(gateway, 'model', None)
        return model if isinstance(model, str) else 'configured gateway'

    @staticmethod
    def _request(gateway, system, prompt, max_tokens):
        # Test and application adapters may implement the original three-argument API.
        if isinstance(gateway, GatewayGenerator):
            return gateway.request_json(system, prompt, max_tokens=max_tokens, timeout_seconds=180)
        return gateway.request_json(system, prompt, max_tokens=max_tokens)

    def prepare_assets(self, prompt):
        prompt = validate_prompt(prompt)
        self.progress(f'Designing original props with {self._model_label(self.gateway)}…')
        system = asset_system_prompt()
        request_system = system
        for attempt in range(2):
            doc = None
            try:
                doc = self._request(self.gateway, request_system, prompt, 16000)
                if not isinstance(doc, dict) or set(doc) != {'assets'} or not doc['assets']:
                    raise ValueError('Expected a nonempty assets pack')
                assets = fit_generated_assets(doc['assets'])
                assets = self._review_assets(assets)
                break
            except ValueError as exc:
                # Transport and HTTP failures need a new user request, while a
                # malformed response or invalid recipe gets one bounded repair.
                if attempt or str(exc).startswith(('Object gateway connection failed', 'Object gateway returned HTTP')):
                    raise
                self.progress('Refining generated prop geometry…')
                previous = ('. Previous pack: ' + json.dumps(doc)) if doc is not None else ''
                request_system = (system + '\nThe previous response failed: ' + str(exc) + previous +
                                  '. Return a complete valid JSON pack. Keep the geometry concise: 6 to 8 assets, '
                                  '8 to 18 parts each, under 180 expanded shapes per asset. Keep distinct window '
                                  'panes with spacing and place facade details in front of opaque walls.')
        self.progress('Saving reviewed props for reuse…')
        self.prepared_assets = self.library.save(assets)
        self.progress(f'Prepared {len(self.prepared_assets)} reusable custom props.')
        return copy.deepcopy(self.prepared_assets)

    def generate(self, prompt):
        prompt = validate_prompt(prompt)
        assets = self.prepared_assets if self.prepared_assets is not None else self.prepare_assets(prompt)
        if not assets:
            raise ValueError('Prepare or select at least one asset before composing a scene')
        from scene_asset_composition import compose_architectural_scene
        architectural = compose_architectural_scene(prompt, assets)
        if architectural is not None:
            self.progress('Custom props composed into an architectural set.')
            return architectural
        self.progress(f'Composing the environment with {self._model_label(self.layout_gateway)}…')
        system = layout_system_prompt(assets)
        doc = self._request(self.layout_gateway, system, prompt, 10000)
        for attempt in range(2):
            try:
                if not isinstance(doc, dict):
                    raise ValueError('Expected a scene object')
                doc = fit_generated_layout(dict(doc, assets=copy.deepcopy(assets)))
                scene = validate_generated_scene(doc)
                if not any(o['kind'] == 'custom' for o in scene['objects']):
                    raise ValueError('Scene must use the prepared custom props')
                self.progress('Scene and custom geometry ready.')
                return scene
            except ValueError as exc:
                if attempt:
                    raise
                self.progress('Refining the scene layout…')
                doc = self._request(self.layout_gateway, system + '\nFix this validation error: '+str(exc)+'. Previous layout: '+json.dumps(doc)+'. Return the full scene.', prompt, 10000)
