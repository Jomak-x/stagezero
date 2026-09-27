"""Reusable, navigable city block set for cinematic character demonstrations.

All geometry is local procedural data.  The interaction targets are suggestions
for future motion tooling and are deliberately kept outside the scene schema.
"""

from __future__ import annotations

import random

from scene_composition import validate_scene
from scene_environments import _asset, _object, _part


_BRICK = (142, 82, 70)
_LIMESTONE = (184, 172, 147)
_SANDSTONE = (175, 143, 105)
_SLATE = (68, 82, 93)
_GLASS = (79, 135, 157)
_NIGHT_GLASS = (48, 76, 95)
_STEEL = (69, 79, 86)


def _building_asset(identifier, name, wall, trim, glass, *,
                    columns=3, rows=7, tower=False, escape=False,
                    billboard=False, stepped=False):
    """Make a complete four-sided facade and a useful, detailed roof."""
    parts = [
        _part('box', (0, -.055, 0), (.89, .87, .87), wall),
        _part('box', (0, -.485, 0), (.96, .03, .96), trim),
        _part('box', (0, .384, 0), (.96, .038, .96), trim),
        _part('box', (0, .407, 0), (.85, .018, .85), (92, 96, 98)),
        _part('box', (0, .427, -.44), (.95, .026, .045), trim),
        _part('box', (0, .427, .44), (.95, .026, .045), trim),
        _part('box', (-.44, .427, 0), (.045, .026, .86), trim),
        _part('box', (.44, .427, 0), (.045, .026, .86), trim),
        _part('box', (0, -.375, .445), (.21, .21, .017), _NIGHT_GLASS),
        _part('box', (0, -.375, -.445), (.21, .21, .017), _NIGHT_GLASS),
        _part('box', (-.445, -.375, 0), (.017, .21, .21), _NIGHT_GLASS),
        _part('box', (.445, -.375, 0), (.017, .21, .21), _NIGHT_GLASS),
    ]
    # Four decorated faces are critical when the camera or actor travels around
    # a whole block.  The repeated windows keep the file compact.
    dx = .24 if columns == 3 else .185
    start = -(columns - 1) * dx / 2
    dy = .095 if rows == 7 else .081
    y0 = -.265
    for face in (-1, 1):
        z = face * .449
        parts.extend([
            _part('box', (start, y0, z), (.155, .059, .012), trim,
                  repeat=((columns, rows, 1), (dx, dy, 0))),
            _part('box', (start, y0, face * .458), (.128, .047, .012), glass,
                  repeat=((columns, rows, 1), (dx, dy, 0))),
            _part('box', (start, y0-.034, face * .465), (.177, .01, .04), trim,
                  repeat=((columns, rows, 1), (dx, dy, 0))),
        ])
        x = face * .449
        parts.extend([
            _part('box', (x, y0, start), (.012, .059, .155), trim,
                  repeat=((1, rows, columns), (0, dy, dx))),
            _part('box', (face * .458, y0, start), (.012, .047, .128), glass,
                  repeat=((1, rows, columns), (0, dy, dx))),
            _part('box', (face * .465, y0-.034, start), (.04, .01, .177), trim,
                  repeat=((1, rows, columns), (0, dy, dx))),
        ])
        if columns == 3:
            parts.extend([
                _part('box', (start, y0, face * .468), (.008, .047, .008), trim,
                      repeat=((columns, rows, 1), (dx, dy, 0))),
                _part('box', (face * .468, y0, start), (.008, .047, .008), trim,
                      repeat=((1, rows, columns), (0, dy, dx))),
            ])
    # Roof plant, service box, vents and parapets form usable anchor/landing
    # landmarks without changing the parent object's rectangular hit area.
    parts.extend([
        _part('box', (-.21, .432, -.19), (.19, .055, .2), _STEEL),
        _part('box', (-.21, .466, -.19), (.21, .013, .22), (128, 133, 126)),
        _part('cylinder', (.22, .441, -.18), (.065, .067, .065), (98, 104, 106)),
        _part('box', (.1, .418, .17), (.28, .03, .12), (82, 87, 89)),
    ])
    if tower:
        parts.extend([
            _part('box', (-.21, .445, -.18), (.02, .085, .02), _STEEL,
                  repeat=((2, 1, 2), (.42, 0, .36))),
            _part('cylinder', (0, .453, 0), (.32, .065, .30), (111, 75, 59)),
            _part('cone', (0, .496, 0), (.35, .008, .32), (73, 80, 84)),
        ])
    if escape:
        parts.extend([
            _part('box', (.464, -.25, .18), (.065, .016, .30), _STEEL,
                  repeat=((1, 4, 1), (0, .19, 0))),
            _part('box', (.485, -.25, .32), (.018, .075, .018), _STEEL,
                  repeat=((1, 4, 1), (0, .19, 0))),
            _part('box', (.47, -.25, .02), (.025, .17, .015), _STEEL,
                  repeat=((1, 4, 1), (0, .19, 0))),
        ])
    if billboard:
        parts.extend([
            _part('box', (0, .465, 0), (.50, .065, .045), (230, 183, 83)),
            _part('box', (0, .468, .031), (.38, .026, .008), (68, 82, 91)),
        ])
    if stepped:
        parts.extend([
            _part('box', (0, .431, -.15), (.48, .07, .42), wall),
            _part('box', (0, .47, -.15), (.52, .014, .46), trim),
        ])
    return _asset(identifier, name, parts)


def _assets():
    return [
        _asset('swing-ground', 'City foundation', [
            _part('box', (0, 0, 0), (1, 1, 1), (111, 111, 107)),
        ]),
        _asset('swing-road', 'Asphalt and lane markings', [
            _part('box', (0, -.015, 0), (1, .94, 1), (48, 54, 61)),
            _part('box', (0, .48, -.43), (.022, .02, .033), (215, 179, 91),
                  repeat=((1, 1, 12), (0, 0, .078))),
            _part('box', (-.39, .48, 0), (.013, .02, .94), (160, 164, 164)),
            _part('box', (.39, .48, 0), (.013, .02, .94), (160, 164, 164)),
        ]),
        _asset('swing-crossing', 'Four-way marked crossing', [
            _part('box', (0, -.015, 0), (1, .94, 1), (52, 57, 62)),
            _part('box', (-.32, .48, -.36), (.065, .012, .15), (230, 224, 204),
                  repeat=((6, 1, 2), (.128, 0, .72))),
            _part('box', (-.36, .48, -.32), (.15, .012, .065), (230, 224, 204),
                  repeat=((2, 1, 6), (.72, 0, .128))),
        ]),
        _asset('swing-plaza', 'Central stone performance plaza', [
            _part('box', (0, -.015, 0), (1, .94, 1), (171, 162, 143)),
            _part('box', (-.36, .49, 0), (.008, .012, .72), (125, 120, 109),
                  repeat=((5, 1, 1), (.18, 0, 0))),
            _part('box', (0, .49, -.36), (.72, .012, .008), (125, 120, 109),
                  repeat=((1, 1, 5), (0, 0, .18))),
        ]),
        _asset('swing-lamp', 'Twin head boulevard light', [
            _part('cylinder', (0, -.47, 0), (.20, .06, .20), _STEEL),
            _part('cylinder', (0, -.01, 0), (.05, .88, .05), _STEEL),
            _part('box', (0, .36, 0), (.66, .035, .055), _STEEL),
            _part('box', (-.27, .29, 0), (.15, .105, .13), (248, 214, 149),
                  repeat=((2, 1, 1), (.54, 0, 0))),
            _part('box', (-.27, .355, 0), (.19, .026, .17), _SLATE,
                  repeat=((2, 1, 1), (.54, 0, 0))),
        ]),
        _asset('swing-bridge', 'Steel pedestrian roof bridge', [
            _part('box', (0, -.25, 0), (1, .12, .45), (93, 104, 108)),
            _part('box', (0, -.12, -.22), (.99, .16, .035), _STEEL),
            _part('box', (0, -.12, .22), (.99, .16, .035), _STEEL),
            _part('box', (-.4, -.08, -.22), (.035, .2, .035), _STEEL,
                  repeat=((5, 1, 2), (.2, 0, .44))),
            _part('box', (0, .02, -.22), (1, .025, .045), (145, 155, 155)),
            _part('box', (0, .02, .22), (1, .025, .045), (145, 155, 155)),
        ]),
        _building_asset('swing-redbrick', 'Red brick tenement', _BRICK,
                        (196, 174, 145), (83, 133, 149), escape=True),
        _building_asset('swing-limestone', 'Limestone apartments', _LIMESTONE,
                        (209, 203, 180), (70, 122, 145), stepped=True),
        _building_asset('swing-glass', 'Blue glass office tower', (64, 91, 106),
                        (125, 149, 159), _GLASS, columns=4),
        _building_asset('swing-dark', 'Dark Art Deco tower', (79, 83, 82),
                        (179, 159, 118), (78, 111, 121), columns=4,
                        stepped=True),
        _building_asset('swing-water', 'Water tower loft', (152, 111, 88),
                        (190, 176, 152), (72, 114, 132), tower=True, escape=True),
        _building_asset('swing-sign', 'Cinema corner building', _SANDSTONE,
                        (203, 188, 154), (79, 120, 137), billboard=True),
        _building_asset('swing-slate', 'Slate and copper offices', _SLATE,
                        (155, 164, 160), (87, 133, 150), columns=4,
                        billboard=True),
    ]


def make_swing_city(seed: int = 0) -> dict:
    """Build a four-way, multi-block city for swing, rooftop and street demos."""
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError('Seed must be an integer from 0 to 1000000')
    rng = random.Random(seed)
    assets = _assets()
    objects = []

    def place(identifier, name, asset, x, y, z, sx, sy, sz, *, yaw=None):
        objects.append(_object(identifier, name, asset, (x, y, z),
                               (sx, sy, sz), yaw=yaw))

    # Four foundation slabs cover 72 × 72 m while each stays under the 60 m
    # per-object size limit.  Their top is below the roadway and building feet.
    for ix, x in enumerate((-18, 18)):
        for iz, z in enumerate((-18, 18)):
            place(f'city-ground-{ix}-{iz}', 'City foundation', 'swing-ground',
                  x, -.14, z, 36, .25, 36)
    for axis in ('x', 'z'):
        for side, coordinate in enumerate((-12, 12)):
            for section, along in enumerate((-18, 18)):
                if axis == 'x':
                    place(f'city-avenue-{axis}-{side}-{section}', 'North–south avenue',
                          'swing-road', coordinate, .015, along, 7.4, .08, 36)
                else:
                    place(f'city-avenue-{axis}-{side}-{section}', 'East–west avenue',
                          'swing-road', along, .015, coordinate, 7.4, .08, 36, yaw=90)
    # A slightly raised intersection hides the coincident road slabs and adds
    # crosswalks at each of the four junctions.
    for ix, x in enumerate((-12, 12)):
        for iz, z in enumerate((-12, 12)):
            place(f'city-crossing-{ix}-{iz}', 'Marked street crossing',
                  'swing-crossing', x, .045, z, 7.4, .08, 7.4)
    place('city-central-plaza', 'Central open plaza', 'swing-plaza',
          0, .015, 0, 15.8, .08, 15.8)

    styles = ('swing-redbrick', 'swing-limestone', 'swing-glass',
              'swing-water', 'swing-dark', 'swing-sign', 'swing-slate')
    # Each of the eight outer blocks has four roofs.  The open central block
    # provides a camera and actor staging area and keeps the crossing readable.
    for bx in (-1, 0, 1):
        for bz in (-1, 0, 1):
            if bx == bz == 0:
                continue
            for ix in (-1, 1):
                for iz in (-1, 1):
                    x = bx * 24 + ix * 4.15
                    z = bz * 24 + iz * 4.15
                    style = styles[(bx * 9 + bz * 5 + ix * 3 + iz + seed) % len(styles)]
                    # Three clear height bands, with variation within every
                    # block, give good anchor heights and a stepped skyline.
                    height = [14, 18, 22, 27, 32, 35][
                        (abs(bx) * 2 + abs(bz) * 3 + (ix == iz) + rng.randrange(6)) % 6]
                    if (bx, bz, iz) in ((-1, -1, -1), (1, 1, 1)):
                        height = 18  # aligned roof pairs for the service bridges
                        style = 'swing-redbrick' if bx < 0 else 'swing-limestone'
                    width = rng.choice((6.7, 7.0, 7.3))
                    depth = rng.choice((6.6, 7.0, 7.2))
                    ident = f'city-building-{bx+1}-{bz+1}-{ix+1}-{iz+1}'
                    place(ident, f'{style.removeprefix("swing-").title()} building',
                          style, x, height / 2, z, width, height, depth)

    # Street furniture marks the avenues without consuming the central area.
    for i, (x, z) in enumerate(((-15.8, -4), (-8.2, 4), (8.2, -4), (15.8, 4),
                                 (-15.8, 20), (-8.2, -20), (8.2, 20), (15.8, -20))):
        place(f'city-lamp-{i}', 'Boulevard light', 'swing-lamp',
              x, 2.3, z, .7, 4.6, .7)
    # Match each bridge deck to the two roofs it joins after the renderer's
    # actual-bounds normalization, which varies a little by building style.
    from asset_geometry import compile_asset

    asset_lookup = {asset['id']: asset for asset in assets}
    bounds = {}

    def normalized_y(asset_id, raw_y):
        if asset_id not in bounds:
            vertices, _, _ = compile_asset(asset_lookup[asset_id])
            bounds[asset_id] = (vertices.min(axis=0), vertices.max(axis=0))
        lower, upper = bounds[asset_id]
        return float((raw_y - (lower[1] + upper[1]) / 2) /
                     (upper[1] - lower[1]))

    bridge_deck_y = normalized_y('swing-bridge', -.19)
    for i, (x, z) in enumerate(((-24, -28.15), (24, 28.15))):
        hosts = [obj for obj in objects if obj['id'].startswith('city-building-')
                 and abs(obj['position'][2] - z) < .001
                 and abs(abs(obj['position'][0] - x) - 4.15) < .001]
        assert len(hosts) == 2
        roof_y = sum(obj['position'][1] + obj['size'][1] *
                     normalized_y(obj['asset'], .416) for obj in hosts) / 2
        place(f'city-bridge-{i}', 'Rooftop service bridge', 'swing-bridge',
              x, roof_y - 1.1 * bridge_deck_y, z, 9.0, 1.1, 2.2)

    scene = {'version': 3, 'name': 'Cinematic swing city', 'assets': assets,
             'objects': objects, 'effects': [], 'lighting': 'sunset',
             'camera': {'position': [54, 48, 66], 'look_at': [0, 11, 0]}}
    return validate_scene(scene)


def suggested_targets(scene: dict) -> list[dict]:
    """Return stable local attachment points for later motion or interaction demos.

    Coordinates are normalized within each referenced object's local box.
    They are scene metadata, not scene objects or active motion constraints.
    """
    from asset_geometry import compile_asset

    assets = {asset['id']: asset for asset in scene['assets']}
    bounds = {}

    def local(asset_id, raw):
        if asset_id not in bounds:
            vertices, _, _ = compile_asset(assets[asset_id])
            bounds[asset_id] = (vertices.min(axis=0), vertices.max(axis=0))
        lower, upper = bounds[asset_id]
        return [float((raw[i] - (lower[i] + upper[i]) / 2) /
                      (upper[i] - lower[i])) for i in range(3)]

    targets = []
    for obj in scene['objects']:
        if not obj['id'].startswith('city-building-'):
            continue
        ident = obj['id']
        roof = obj['name'].replace(' building', '')
        asset_id = obj['asset']
        targets.extend((
            {'id': f'{ident}-landing', 'name': f'{roof} roof landing',
             'object_id': ident, 'kind': 'landing',
             'local_position': local(asset_id, (0, .416, .27))},
            {'id': f'{ident}-anchor', 'name': f'{roof} parapet swing anchor',
             'object_id': ident, 'kind': 'swing_anchor',
             'local_position': local(asset_id, (.44, .44, .4))},
            {'id': f'{ident}-climb', 'name': f'{roof} wall climb',
             'object_id': ident, 'kind': 'climb',
             'local_position': local(asset_id, (.472, 0, 0))},
        ))
    for obj in scene['objects']:
        if obj['asset'] == 'swing-bridge':
            targets.append({'id': f'{obj["id"]}-vault', 'name': 'Bridge vault',
                            'object_id': obj['id'], 'kind': 'vault',
                            'local_position': local(obj['asset'], (0, .032, .22))})
    return targets
