"""Detailed, deterministic scene sets built from reusable local geometry.

The blueprints are ordinary version 3 scene assets.  They require no downloaded
meshes and remain editable as scene objects after a preset is created.
"""

from __future__ import annotations

import random


def _part(shape, position, size, color, *, repeat=None, rotation=None):
    part = {"shape": shape, "position": list(position), "size": list(size), "color": list(color)}
    if repeat is not None:
        count, step = repeat
        part["repeat"] = {"count": list(count), "step": list(step)}
    if rotation is not None:
        part["rotation"] = list(rotation)
    return part


def _asset(identifier, name, parts):
    return {"id": identifier, "name": name, "parts": parts}


def _object(identifier, name, asset, position, size, color=(255, 255, 255), *, yaw=None):
    result = {"id": identifier, "name": name, "kind": "custom", "asset": asset,
              "position": list(position), "size": list(size), "color": list(color),
              "interaction": {"action": "none", "trigger": "none", "radius": 0}}
    if yaw is not None:
        result["yaw"] = yaw
    return result


def _building(identifier, name, wall, trim, glass, accent, *, columns=3, rows=4, office=False):
    """A facade with actual window frames, mullions, cornice and ground floor."""
    parts = [
        _part("box", (0, -.025, 0), (.92, .90, .76), wall),
        _part("box", (0, .445, .02), (.99, .055, .84), trim),
        _part("box", (0, -.465, .39), (.96, .055, .045), trim),
        _part("box", (0, -.35, .398), (.91, .17, .025), accent),
        _part("box", (0, -.235, .407), (.95, .035, .035), trim),
        _part("box", (0, -.405, .425), (.2, .18, .012), (38, 43, 51)),
        _part("box", (0, -.405, .438), (.015, .17, .012), trim),
        _part("box", (0, -.47, .45), (.26, .025, .09), trim),
    ]
    if office:
        # Close horizontal bands make a glass office distinct from the masonry blocks.
        parts.extend([
            _part("box", (0, .01, .395), (.85, .70, .018), glass),
            _part("box", (-.29, -.05, .414), (.018, .69, .025), trim,
                  repeat=((3, 1, 1), (.29, 0, 0))),
            _part("box", (0, -.18, .418), (.87, .018, .026), trim,
                  repeat=((1, 4, 1), (0, .17, 0))),
            _part("box", (-.36, -.16, .427), (.1, .05, .008), (124, 180, 193),
                  repeat=((4, 3, 1), (.21, .18, 0))),
            _part("box", (.25, .485, -.05), (.24, .03, .32), (58, 68, 75)),
        ])
    else:
        spacing_x = .26 if columns == 3 else .20
        start_x = -(columns - 1) * spacing_x / 2
        spacing_y = .157 if rows == 4 else .126
        start_y = -.18
        repeat = ((columns, rows, 1), (spacing_x, spacing_y, 0))
        parts.extend([
            _part("box", (start_x, start_y, .396), (.174, .111, .02), trim, repeat=repeat),
            _part("box", (start_x, start_y, .412), (.145, .085, .012), glass, repeat=repeat),
            _part("box", (start_x, start_y-.062, .425), (.19, .018, .055), trim, repeat=repeat),
            _part("box", (start_x, start_y, .422), (.009, .083, .012), trim, repeat=repeat),
            _part("box", (-.448, 0, .389), (.023, .88, .034), trim,
                  repeat=((2, 1, 1), (.896, 0, 0))),
            _part("box", (0, .465, -.04), (.20, .035, .33), (69, 73, 71)),
        ])
    return _asset(identifier, name, parts)


def _street_assets(palette):
    brick, stone, glass, awning, foliage = palette
    asphalt = _asset("asphalt", "Asphalt with lane markings", [
        _part("box", (0, -.015, 0), (1, .94, 1), (49, 54, 60)),
        _part("box", (0, .49, -.32), (.018, .012, .13), (217, 185, 106),
              repeat=((1, 1, 5), (0, 0, .16))),
        _part("box", (-.30, .49, -.42), (.008, .012, .14), (174, 177, 171),
              repeat=((2, 1, 6), (.60, 0, .14))),
        _part("box", (-.28, .49, -.12), (.073, .012, .034), (231, 226, 207),
              repeat=((6, 1, 1), (.11, 0, 0))),
    ])
    sidewalk = _asset("sidewalk", "Stone sidewalk and curb", [
        _part("box", (0, -.015, 0), (1, .94, 1), (161, 158, 147)),
        _part("box", (-.43, .49, 0), (.12, .014, 1), stone),
        _part("box", (-.27, .49, -.36), (.008, .012, .28), (119, 119, 113),
              repeat=((3, 1, 3), (.34, 0, .36))),
    ])
    lamp = _asset("street-lamp", "Cast iron street light", [
        _part("cylinder", (0, -.45, 0), (.21, .05, .21), (50, 59, 62)),
        _part("cylinder", (0, -.02, 0), (.055, .84, .055), (49, 58, 62)),
        _part("box", (0, .32, 0), (.31, .025, .06), (47, 57, 62)),
        _part("box", (-.13, .245, 0), (.11, .14, .11), (229, 201, 143),
              repeat=((2, 1, 1), (.26, 0, 0))),
        _part("cone", (-.13, .34, 0), (.15, .07, .15), (58, 68, 70),
              repeat=((2, 1, 1), (.26, 0, 0))),
    ])
    tree = _asset("street-tree", "Planter tree", [
        _part("box", (0, -.42, 0), (.28, .15, .28), stone),
        _part("cylinder", (0, -.19, 0), (.09, .40, .09), (88, 68, 51)),
        _part("sphere", (0, .22, 0), (.58, .52, .51), foliage),
        _part("sphere", (-.21, .04, .04), (.29, .30, .33), (57, 104, 75)),
        _part("sphere", (.19, .12, -.1), (.33, .36, .32), (65, 121, 83)),
    ])
    bench = _asset("city-bench", "Timber and iron bench", [
        _part("box", (0, -.03, 0), (.82, .11, .35), (120, 82, 53)),
        _part("box", (0, .18, -.16), (.82, .35, .07), (130, 91, 60)),
        _part("box", (-.34, -.29, 0), (.05, .34, .30), (56, 62, 62),
              repeat=((2, 1, 1), (.68, 0, 0))),
        _part("box", (0, .30, -.11), (.86, .035, .07), (81, 61, 45)),
    ])
    storefront = _asset("storefront", "Corner cafe storefront", [
        _part("box", (0, 0, 0), (.94, .86, .76), brick),
        _part("box", (0, -.38, .4), (.9, .20, .022), (45, 65, 68)),
        _part("box", (-.31, -.37, .419), (.24, .15, .012), (113, 166, 171),
              repeat=((3, 1, 1), (.31, 0, 0))),
        _part("box", (0, -.28, .45), (.97, .035, .09), awning),
        _part("box", (-.385, -.30, .485), (.12, .035, .012), (241, 232, 209),
              repeat=((5, 1, 1), (.19, 0, 0))),
        _part("box", (0, -.16, .401), (.88, .09, .03), stone),
        _part("box", (0, .16, .401), (.87, .54, .02), (44, 64, 72)),
        _part("box", (-.29, .15, .417), (.23, .43, .012), glass,
              repeat=((3, 1, 1), (.29, 0, 0))),
        _part("box", (0, .455, 0), (1, .08, .83), stone),
    ])
    return [_asset("city-lot", "Paved city ground", [_part("box", (0,0,0), (1,1,1), (115,117,113))]), asphalt, sidewalk,
            _building("masonry", "Brick apartment facade", brick, stone, glass, awning),
            _building("office", "Glass office facade", (68, 80, 87), (111, 124, 128),
                      (57, 103, 128), (49, 67, 71), office=True),
            _building("brownstone", "Stone townhouse facade", (127, 105, 87),
                      (187, 166, 137), (73, 107, 127), (83, 67, 64), columns=3, rows=4),
            storefront, lamp, tree, bench]


def make_city(seed=0):
    """A continuous pair of urban blocks around a clear performance street."""
    rng = random.Random(seed)
    palettes = [
        ((137, 91, 79), (188, 181, 160), (71, 111, 135), (166, 74, 65), (67, 118, 81)),
        ((118, 112, 105), (192, 187, 170), (89, 128, 148), (64, 112, 130), (71, 116, 84)),
        ((153, 111, 79), (203, 175, 141), (82, 117, 138), (172, 112, 57), (74, 119, 84)),
    ]
    palette = palettes[seed % len(palettes)]
    assets = _street_assets(palette)
    objects = []

    def place(name, asset, x, y, z, sx, sy, sz, *, yaw=None):
        objects.append(_object(f"city-{len(objects)}", name, asset, (x, y, z), (sx, sy, sz), yaw=yaw))

    place("City ground", "city-lot", 0, -.015, -12.5, 27, .05, 45)
    place("Main street", "asphalt", 0, -.015, -12, 8.2, .08, 41)
    for side in (-1, 1):
        place("Sidewalk", "sidewalk", side*4.78, -.02, -12, 1.35, .12, 41,
              yaw=180 if side == 1 else None)
    # The facade recipe looks toward local +Z. Rotate each row so its shopfronts
    # face the avenue, rather than presenting blank side walls to the camera.
    for side in (-1, 1):
        for index in range(9):
            asset = ("masonry", "brownstone", "office", "storefront")[(index + (side == 1)) % 4]
            height = (6.4, 5.2, 8.5, 3.8)[("masonry", "brownstone", "office", "storefront").index(asset)]
            height += rng.choice((-.3, 0, .3))
            place("Streetfront building", asset, side*6.85, height/2,
                  -2.5-index*3.2, 3.08, height, 2.55,
                  yaw=90 if side == -1 else -90)
        for z in (-1.2, -7.6, -14, -20.4, -26.8):
            place("Street lamp", "street-lamp", side*4.25, 1.45, z, .55, 2.9, .55)
        for z in (-4.3, -10.7, -17.1, -23.5, -29.9):
            place("Planter tree", "street-tree", side*5.18, 1.35, z, 1.25, 2.7, 1.25)
    # A second, taller depth layer closes the view at the far intersection.
    for index, x in enumerate((-11.9, -7.9, -3.9, 3.9, 7.9, 11.9)):
        asset = ("office", "brownstone", "masonry")[index % 3]
        height = (11.8, 10.2, 13.3, 12.6, 10.8, 14.1)[index]
        place("Distant skyline building", asset, x, height/2,
              -33.1-rng.uniform(0, .3), 3.8, height, 3.0)
    place("Sidewalk bench", "city-bench", -4.92, .45, -5.7, 1.4, .9, .72)
    place("Sidewalk bench", "city-bench", 4.92, .45, -18.4, 1.4, .9, .72)
    return {"version": 3, "name": "Downtown street", "assets": assets,
            "objects": objects, "effects": [], "lighting": "sunset" if seed % 3 == 2 else "neutral",
            "camera": {"position": [4.0, 3.1, 10.5], "look_at": [0, 2.1, -5.5]}}


def _house(identifier, name, siding, roof, trim):
    """Compact pitched-roof home with a front porch and street-facing windows."""
    parts = [
        _part("box", (0, -.155, 0), (.82, .66, .72), siding),
        _part("box", (-.205, .275, 0), (.52, .085, .80), roof,
              rotation=(0, 0, 30)),
        _part("box", (.205, .275, 0), (.52, .085, .80), roof,
              rotation=(0, 0, -30)),
        _part("box", (0, .43, 0), (.07, .05, .82), trim),
        _part("box", (-.39, -.145, .365), (.035, .61, .035), trim,
              repeat=((2, 1, 1), (.78, 0, 0))),
        _part("box", (0, -.105, .369), (.72, .022, .035), trim),
        _part("box", (0, -.385, .369), (.82, .035, .06), trim),
        _part("box", (0, -.32, .386), (.16, .31, .014), (67, 86, 90)),
        _part("box", (0, -.32, .398), (.022, .31, .016), trim),
        _part("sphere", (.05, -.32, .409), (.024, .024, .024), (215, 182, 95)),
        _part("box", (-.255, -.215, .38), (.20, .24, .022), trim,
              repeat=((2, 1, 1), (.51, 0, 0))),
        _part("box", (-.255, -.215, .395), (.16, .19, .015), (98, 159, 176),
              repeat=((2, 1, 1), (.51, 0, 0))),
        _part("box", (-.255, -.215, .408), (.012, .20, .018), trim,
              repeat=((2, 1, 1), (.51, 0, 0))),
        _part("box", (-.255, -.345, .408), (.23, .018, .055), trim,
              repeat=((2, 1, 1), (.51, 0, 0))),
        _part("box", (0, -.455, .405), (.35, .055, .15), (165, 159, 143)),
        _part("box", (0, -.445, .49), (.25, .025, .018), trim),
        _part("box", (.31, .375, -.16), (.12, .21, .13), siding),
        _part("box", (.31, .49, -.16), (.15, .02, .16), roof),
    ]
    return _asset(identifier, name, parts)


def _residential_assets(palette):
    brick, stone, glass, awning, foliage = palette
    street = {asset["id"]: asset for asset in _street_assets(palette)}
    return [
        _asset("residential-ground", "Neighborhood ground", [
            _part("box", (0, 0, 0), (1, 1, 1), (91, 128, 82))]),
        _asset("neighborhood-road", "Residential asphalt road", [
            _part("box", (0, -.015, 0), (1, .94, 1), (69, 73, 76)),
            _part("box", (0, .49, -.43), (.018, .012, .11), (235, 212, 135),
                  repeat=((1, 1, 7), (0, 0, .14))),
            _part("box", (-.47, .49, 0), (.012, .012, .96), (216, 213, 195),
                  repeat=((2, 1, 1), (.94, 0, 0))),
        ]),
        street["sidewalk"],
        _asset("residential-yard", "Front lawns", [
            _part("box", (0, -.015, 0), (1, .94, 1), (101, 149, 91)),
            _part("box", (-.45, .49, -.47), (.08, .012, .06), (76, 125, 78),
                  repeat=((10, 1, 8), (.1, 0, .13))),
        ]),
        _house("house-clapboard", "Blue clapboard home", (117, 153, 166),
               (82, 91, 104), (230, 226, 207)),
        _house("house-brick", "Warm brick home", (157, 102, 83),
               (94, 75, 71), (232, 217, 185)),
        _asset("residential-fence", "Picket fence with gate opening", [
            _part("box", (-.305, -.14, 0), (.32, .04, .055), (227, 224, 202),
                  repeat=((2, 2, 1), (.61, .25, 0))),
            _part("box", (-.44, -.12, 0), (.045, .74, .09), (231, 226, 208),
                  repeat=((2, 1, 1), (.88, 0, 0))),
            _part("box", (-.445, -.15, 0), (.035, .60, .05), (239, 236, 219),
                  repeat=((4, 1, 1), (.095, 0, 0))),
            _part("box", (.16, -.15, 0), (.035, .60, .05), (239, 236, 219),
                  repeat=((4, 1, 1), (.095, 0, 0))),
        ]),
        _asset("mailbox", "Curbside mailbox", [
            _part("box", (0, -.22, 0), (.07, .56, .07), (94, 85, 75)),
            _part("box", (0, .14, 0), (.34, .22, .27), (59, 93, 115)),
            _part("box", (0, .275, 0), (.36, .045, .29), (65, 100, 121)),
            _part("box", (.19, .16, 0), (.04, .15, .055), (184, 71, 59)),
        ]),
        street["street-lamp"], street["street-tree"],
    ]


def make_residential(seed=0):
    """Two lived-in rows of houses, lawns and fences along a quiet road."""
    rng = random.Random(seed)
    palettes = [
        ((157, 102, 83), (185, 180, 164), (98, 159, 176), (143, 91, 70), (78, 136, 79)),
        ((139, 106, 83), (202, 191, 170), (83, 141, 165), (98, 121, 142), (71, 124, 74)),
        ((167, 120, 93), (197, 184, 154), (96, 151, 170), (142, 97, 80), (91, 140, 75)),
    ]
    assets = _residential_assets(palettes[seed % len(palettes)])
    objects = []

    def place(name, asset, x, y, z, sx, sy, sz, *, yaw=None):
        objects.append(_object(f"residential-{len(objects)}", name, asset,
                               (x, y, z), (sx, sy, sz), yaw=yaw))

    place("Neighborhood ground", "residential-ground", 0, -.015, -12.5, 25, .05, 45)
    place("Quiet neighborhood road", "neighborhood-road", 0, -.015, -12, 8.1, .08, 41)
    for side in (-1, 1):
        place("Sidewalk", "sidewalk", side*4.78, -.02, -12, 1.3, .12, 41,
              yaw=180 if side == 1 else None)
        place("Front lawns", "residential-yard", side*8.4, -.018, -12.5,
              5.9, .10, 39)
        for index in range(8):
            z = -2.5-index*4.0
            house = "house-clapboard" if (index + (side == 1) + seed) % 3 else "house-brick"
            height = 3.7 + rng.choice((-.12, 0, .12))
            place("Neighborhood house", house, side*8.3, height/2, z,
                  3.25, height, 2.9, yaw=90 if side == -1 else -90)
            place("Front picket fence", "residential-fence", side*6.25, .42, z,
                  3.3, .84, .12, yaw=90 if side == -1 else -90)
            if index in (1, 3, 5, 7):
                place("Yard tree", "street-tree", side*10.55, 1.5,
                      z-.65, 1.55, 3.0, 1.55)
            if index in (0, 3, 6):
                place("Street lamp", "street-lamp", side*4.25, 1.45,
                      z-1.5, .55, 2.9, .55)
            if index in (1, 5):
                place("Curbside mailbox", "mailbox", side*5.65, .55,
                      z+1.05, .48, 1.1, .4)
    return {"version": 3, "name": "Neighborhood street", "assets": assets,
            "objects": objects, "effects": [], "lighting": "warm" if seed % 2 else "neutral",
            "camera": {"position": [3.7, 2.8, 10.5], "look_at": [0, 1.8, -6.0]}}


def _room_assets(palette):
    wall, wood, fabric, trim, leaf = palette
    floor = _asset("floorboards", "Herringbone wood floor", [
        _part("box", (0, -.015, 0), (1, .94, 1), wood),
        _part("box", (-.45, .49, 0), (.012, .012, .96), (92, 66, 48),
              repeat=((10, 1, 1), (.1, 0, 0))),
        _part("box", (-.45, .49, -.45), (.1, .012, .008), (169, 125, 82),
              repeat=((10, 1, 10), (.1, 0, .1))),
    ])
    rear = _asset("rear-wall", "Paneled wall with tall windows", [
        _part("box", (0, 0, 0), (1, 1, .82), wall),
        _part("box", (0, -.43, .44), (1, .11, .045), trim),
        _part("box", (0, -.20, .43), (1, .018, .035), trim),
        _part("box", (0, .45, .43), (1, .07, .055), trim),
        _part("box", (-.315, .14, .425), (.275, .48, .015), (45, 76, 94),
              repeat=((3, 1, 1), (.315, 0, 0))),
        _part("box", (-.315, -.105, .445), (.29, .018, .035), trim,
              repeat=((3, 2, 1), (.315, .49, 0))),
        _part("box", (-.4725, .14, .449), (.016, .50, .035), trim,
              repeat=((4, 1, 1), (.315, 0, 0))),
        _part("box", (-.315, .14, .457), (.013, .45, .012), (173, 199, 199),
              repeat=((3, 1, 1), (.315, 0, 0))),
        _part("box", (-.315, -.07, .46), (.29, .035, .065), trim,
              repeat=((3, 1, 1), (.315, 0, 0))),
        _part("box", (-.34, -.32, .46), (.25, .13, .015), (212, 207, 185),
              repeat=((3, 1, 1), (.34, 0, 0))),
    ])
    side = _asset("side-wall", "Wall with timber wainscot", [
        _part("box", (0, 0, -.015), (1, 1, .94), wall),
        _part("box", (0, -.39, .49), (1, .20, .012), (171, 151, 127)),
        _part("box", (0, -.24, .49), (1, .024, .012), trim),
        _part("box", (-.42, -.38, .49), (.012, .16, .012), trim,
              repeat=((6, 1, 1), (.17, 0, 0))),
        _part("box", (0, .43, .49), (1, .055, .012), trim),
    ])
    rug = _asset("woven-rug", "Bordered woven rug", [
        _part("box", (0, 0, 0), (1, .14, 1), (190, 126, 98)),
        _part("box", (0, .09, 0), (.86, .025, .86), fabric),
        _part("box", (0, .11, 0), (.72, .01, .72), (203, 164, 123)),
        _part("box", (0, .12, 0), (.56, .01, .56), fabric),
        _part("box", (-.44, .04, -.43), (.02, .06, .07), (228, 202, 162),
              repeat=((9, 1, 2), (.11, 0, .86))),
    ])
    couch = _asset("sofa", "Upholstered sofa with cushions", [
        _part("box", (0, -.12, 0), (.96, .29, .83), fabric),
        _part("box", (0, .20, -.34), (.98, .48, .22), fabric),
        _part("box", (-.42, .04, 0), (.15, .54, .83), fabric,
              repeat=((2, 1, 1), (.84, 0, 0))),
        _part("box", (-.21, -.30, 0), (.10, .27, .55), (77, 64, 56),
              repeat=((2, 1, 1), (.42, 0, 0))),
        _part("box", (-.20, .20, -.20), (.32, .24, .14), (222, 187, 140),
              repeat=((2, 1, 1), (.40, 0, 0))),
        _part("box", (0, -.10, .42), (.82, .055, .055), trim),
    ])
    table = _asset("coffee-table", "Low walnut coffee table", [
        _part("box", (0, .24, 0), (.91, .10, .78), wood),
        _part("box", (-.36, -.08, -.29), (.08, .55, .08), (89, 63, 48),
              repeat=((2, 1, 2), (.72, 0, .58))),
        _part("box", (0, -.12, 0), (.76, .05, .62), (108, 78, 55)),
        _part("cylinder", (.20, .325, -.08), (.18, .07, .18), (231, 220, 195)),
        _part("sphere", (.20, .365, -.08), (.08, .06, .08), (164, 113, 72)),
    ])
    shelf = _asset("bookshelf", "Bookcase full of varied books", [
        _part("box", (0, 0, -.30), (.92, .95, .16), wood),
        _part("box", (-.42, 0, 0), (.09, .95, .76), trim,
              repeat=((2, 1, 1), (.84, 0, 0))),
        _part("box", (0, -.44, 0), (.92, .075, .77), trim,
              repeat=((1, 5, 1), (0, .22, 0))),
        _part("box", (-.32, -.32, .20), (.065, .18, .36), (116, 134, 139),
              repeat=((7, 4, 1), (.105, .22, 0))),
        _part("box", (-.27, -.32, .405), (.04, .17, .012), (182, 123, 80),
              repeat=((7, 4, 1), (.105, .22, 0))),
        _part("box", (0, .48, 0), (1, .04, .81), trim),
    ])
    plant = _asset("indoor-plant", "Leafy potted plant", [
        _part("cylinder", (0, -.36, 0), (.34, .25, .34), (158, 102, 70)),
        _part("cylinder", (0, -.12, 0), (.05, .31, .05), (74, 89, 59)),
        _part("sphere", (0, .22, 0), (.47, .43, .44), leaf),
        _part("sphere", (-.20, .04, .10), (.25, .28, .25), (76, 128, 84)),
        _part("sphere", (.18, .13, -.09), (.27, .29, .25), (98, 145, 86)),
    ])
    lamp = _asset("floor-lamp", "Brass floor lamp and pleated shade", [
        _part("cylinder", (0, -.47, 0), (.30, .035, .30), (160, 130, 82)),
        _part("cylinder", (0, -.06, 0), (.035, .79, .035), (181, 148, 88)),
        _part("cone", (0, .36, 0), (.40, .22, .40), (225, 201, 156)),
        _part("cylinder", (0, .47, 0), (.10, .03, .10), (164, 130, 75)),
    ])
    art = _asset("wall-art", "Framed abstract painting", [
        _part("box", (0, 0, -.2), (1, 1, .36), (90, 61, 43)),
        _part("box", (0, 0, .04), (.86, .84, .12), (229, 220, 195)),
        _part("box", (-.15, .08, .12), (.34, .57, .08), (95, 123, 130), rotation=(0, 0, 18)),
        _part("sphere", (.19, -.13, .17), (.33, .29, .09), (192, 134, 91)),
    ])
    return [floor, rear, side, rug, couch, table, shelf, plant, lamp, art]


def make_room(seed=0):
    """Warm open-front living room with architectural detail and furniture."""
    palettes = [
        ((211, 199, 178), (136, 95, 63), (92, 128, 129), (231, 219, 196), (74, 132, 89)),
        ((201, 209, 202), (119, 91, 70), (183, 117, 97), (230, 225, 205), (65, 126, 89)),
        ((212, 198, 185), (150, 104, 70), (137, 119, 151), (233, 219, 199), (76, 135, 93)),
    ]
    palette = palettes[seed % len(palettes)]
    assets = _room_assets(palette)
    objects = []

    def place(name, asset, x, y, z, sx, sy, sz, *, yaw=None):
        objects.append(_object(f"room-{len(objects)}", name, asset, (x, y, z), (sx, sy, sz), yaw=yaw))

    place("Wood floor", "floorboards", 0, -.020, -.4, 8.2, .09, 7.6)
    place("Window wall", "rear-wall", 0, 1.55, -4.12, 8.2, 3.1, .18)
    place("Left wall", "side-wall", -4.06, 1.55, -2.45, 3.4, 3.1, .15, yaw=90)
    place("Right wall", "side-wall", 4.06, 1.55, -3.35, 1.5, 3.1, .15, yaw=-90)
    place("Woven rug", "woven-rug", 0, .015, -.75, 3.9, .06, 2.8)
    shift = (-.15, 0, .15)[seed % 3]
    place("Sofa", "sofa", -2.35+shift, .50, -1.95, 2.25, 1.0, 1.05)
    place("Coffee table", "coffee-table", 1.55+shift, .30, -1.65, 1.22, .60, .85)
    place("Bookshelf", "bookshelf", 3.15, 1.24, -3.13, 1.05, 2.48, .55)
    place("Floor lamp", "floor-lamp", -3.24, .91, -2.95, .55, 1.82, .55)
    place("Potted plant", "indoor-plant", 3.35, .77, -1.38, .85, 1.54, .85)
    place("Framed painting", "wall-art", -2.8, 2.12, -3.96, .66, .75, .08)
    return {"version": 3, "name": "Furnished living room", "assets": assets,
            "objects": objects, "effects": [], "lighting": "warm",
            "camera": {"position": [3.0, 2.25, 5.5], "look_at": [0, 1.05, -1.25]}}
