"""Generate one reusable custom prop and a scene placement from one prompt."""

from adaptive_scene_generation import AdaptiveSceneGenerator, fit_generated_assets
from asset_geometry import validate_assets
from object_generation import GatewayGenerator, validate_prompt
from scene_asset_library import AssetLibrary
from scene_objects import validate_objects


def single_prop_system_prompt():
    return '''Create exactly ONE original stylized 3D object described by the user.
Return JSON only with exactly these keys: {"assets":[{"id":"safe-id","name":"Descriptive name","parts":[...]}],"placement":{"size":[width,height,depth],"position":[x,y,z],"yaw":0}}.
The assets array must have exactly one item. The object must be a single independent prop, not an entire room or scene.
Each part requires exactly shape (box,sphere,cylinder,cone), position [x,y,z], size [width,height,depth], color [R,G,B] integer 0..255. Optional rotation [rx,ry,rz] degrees and repeat {"count":[nx,ny,nz],"step":[dx,dy,dz]} are allowed.
Parts use normalized LOCAL coordinates; every vertex, including rotated or repeated parts, must fit within [-0.5,0.5] on all axes. Y is up. The complete geometry is scaled by placement.size in meters.
Build a recognizable silhouette with useful structure and details. Use roughly 8 to 22 parts, up to 32; avoid intersecting coplanar surfaces and floating details. No external files, textures, scripts, URLs, or extra keys.
Placement is the object's CENTER in meters. Choose plausible real-world dimensions between 0.05 and 6 meters per axis. Put floor-standing objects with centerY=height/2; suspend only objects that naturally hang. Put the object near x=2,z=1.5 by default to leave the actor at the origin clear. Use yaw in degrees. Do not create a ground plane.'''


class SinglePropGenerator:
    """A bounded gateway request whose result can be appended to any scene."""

    def __init__(self, gateway=None, library=None):
        self.gateway = gateway if gateway is not None else GatewayGenerator.from_env(stage='assets')
        self.library = library if library is not None else AssetLibrary()

    def generate(self, prompt):
        prompt = validate_prompt(prompt)
        system = single_prop_system_prompt()
        for attempt in range(2):
            try:
                document = AdaptiveSceneGenerator._request(self.gateway, system, prompt, 7000)
                if not isinstance(document, dict) or set(document) != {'assets', 'placement'}:
                    raise ValueError('Expected one asset and one placement')
                assets = document['assets']
                if not isinstance(assets, list) or len(assets) != 1:
                    raise ValueError('Generate exactly one custom object')
                asset = AdaptiveSceneGenerator._review_assets(fit_generated_assets(assets))[0]
                placement = document['placement']
                if not isinstance(placement, dict) or set(placement) != {'size', 'position', 'yaw'}:
                    raise ValueError('Object placement needs size, position and yaw')
                prototype = {'id': 'generated-prop', 'name': asset['name'], 'kind': 'custom',
                             'asset': asset['id'], 'size': placement['size'],
                             'position': placement['position'], 'yaw': placement['yaw'],
                             'color': [255, 255, 255],
                             'interaction': {'action': 'none', 'trigger': 'none', 'radius': 0}}
                prototype = validate_objects([prototype])[0]
                if max(prototype['size']) > 6:
                    raise ValueError('Object dimensions must stay within 6 meters')
                asset = self.library.save(validate_assets([asset]))[0]
                prototype['asset'] = asset['id']
                return {'asset': asset, 'object': prototype}
            except ValueError as exc:
                if attempt or str(exc).startswith(('Object gateway connection failed', 'Object gateway returned HTTP')):
                    raise
                system += '\nPrevious result failed validation: ' + str(exc) + '. Return a corrected complete single object.'
