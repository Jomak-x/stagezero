"""Large, editable action sets built entirely from bounded procedural assets.

Every prop is an ordinary version 3 scene object. ``suggested_targets`` is a
separate authoring aid: its points are on modeled surfaces in normalized local
coordinates, and it makes no claim about collision or motion physics.
"""

from __future__ import annotations

import random

from scene_objects import make_object


def _part(shape, position, size, color, *, repeat=None, rotation=None):
    part = {"shape": shape, "position": list(position), "size": list(size),
            "color": list(color)}
    if repeat is not None:
        count, step = repeat
        part["repeat"] = {"count": list(count), "step": list(step)}
    if rotation is not None:
        part["rotation"] = list(rotation)
    return part


def _asset(identifier, name, parts):
    return {"id": identifier, "name": name, "parts": parts}


class _Set:
    def __init__(self, prefix):
        self.prefix = prefix
        self.objects = []

    def add(self, name, asset, position, size, *, yaw=None):
        obj = {"id": f"{self.prefix}-{len(self.objects):02d}", "name": name,
               "kind": "custom", "asset": asset, "position": list(position),
               "size": list(size), "color": [255, 255, 255],
               "interaction": {"action": "none", "trigger": "none", "radius": 0}}
        if yaw is not None:
            obj["yaw"] = yaw
        self.objects.append(obj)
        return obj


def _dock_assets():
    steel = (61, 76, 83)
    dark = (36, 53, 61)
    yellow = (227, 179, 68)
    return [
        _asset("dock-ground", "Scored concrete harbor apron", [
            _part("box", (0, -.01, 0), (1, .98, 1), (116, 125, 124)),
            _part("box", (-.42, .493, 0), (.012, .01, .92), (87, 97, 100), repeat=((6, 1, 1), (.165, 0, 0))),
            _part("box", (0, .486, -.40), (.92, .01, .012), (87, 97, 100), repeat=((1, 1, 5), (0, 0, .20))),
        ]),
        _asset("dock-quay", "Quay edge with hazard striping", [
            _part("box", (0, -.09, 0), (1, .82, 1), (105, 112, 111)),
            _part("box", (-.46, .34, 0), (.08, .04, 1), (66, 77, 81)),
            _part("box", (-.38, .34, -.42), (.09, .012, .11), yellow, repeat=((1, 1, 5), (0, 0, .19))),
            _part("box", (-.49, -.12, -.38), (.015, .28, .12), (47, 59, 63), repeat=((1, 1, 5), (0, 0, .19))),
        ]),
        _asset("dock-water", "Deep harbor water", [
            _part("box", (0, -.12, 0), (1, .76, 1), (27, 68, 83)),
            _part("box", (-.32, .27, -.33), (.21, .006, .018), (67, 135, 150), repeat=((4, 1, 4), (.20, 0, .19))),
        ]),
        _asset("dock-container", "Ribbed shipping container", [
            _part("box", (0, 0, 0), (.94, .90, .92), (172, 84, 65)),
            _part("box", (-.43, 0, .464), (.045, .91, .018), (112, 59, 55), repeat=((11, 1, 1), (.086, 0, 0))),
            _part("box", (-.43, 0, -.464), (.045, .91, .018), (112, 59, 55), repeat=((11, 1, 1), (.086, 0, 0))),
            _part("box", (-.47, 0, 0), (.055, .98, .99), steel, repeat=((2, 1, 1), (.94, 0, 0))),
            _part("box", (0, -.47, 0), (.99, .055, .99), steel, repeat=((1, 2, 1), (0, .94, 0))),
            _part("box", (0, 0, .48), (.016, .84, .014), yellow),
            _part("box", (-.22, -.02, .48), (.014, .83, .012), steel, repeat=((2, 1, 1), (.44, 0, 0))),
            _part("box", (-.13, -.18, .488), (.07, .025, .022), (222, 219, 199), repeat=((2, 1, 1), (.26, 0, 0))),
        ]),
        _asset("dock-container-blue", "Blue ribbed shipping container", [
            _part("box", (0, 0, 0), (.94, .90, .92), (53, 111, 133)),
            _part("box", (-.43, 0, .464), (.045, .91, .018), (38, 78, 93), repeat=((11, 1, 1), (.086, 0, 0))),
            _part("box", (-.43, 0, -.464), (.045, .91, .018), (38, 78, 93), repeat=((11, 1, 1), (.086, 0, 0))),
            _part("box", (-.47, 0, 0), (.055, .98, .99), steel, repeat=((2, 1, 1), (.94, 0, 0))),
            _part("box", (0, -.47, 0), (.99, .055, .99), steel, repeat=((1, 2, 1), (0, .94, 0))),
            _part("box", (0, 0, .48), (.016, .84, .014), yellow),
            _part("box", (-.22, -.02, .48), (.014, .83, .012), steel, repeat=((2, 1, 1), (.44, 0, 0))),
        ]),
        _asset("dock-gantry", "Container gantry with elevated crossbeam", [
            _part("box", (-.43, -.05, 0), (.09, .9, .25), steel, repeat=((2, 1, 1), (.86, 0, 0))),
            _part("box", (0, .43, 0), (.98, .13, .31), yellow),
            _part("box", (0, .30, 0), (.87, .025, .22), dark),
            _part("box", (-.37, .28, .16), (.035, .24, .035), steel, repeat=((5, 1, 1), (.185, 0, 0))),
            _part("box", (0, .41, -.19), (.76, .035, .07), dark),
            _part("box", (-.34, .49, -.18), (.015, .015, .015), yellow, repeat=((5, 1, 1), (.17, 0, 0))),
            _part("box", (-.41, -.47, 0), (.17, .05, .40), dark, repeat=((2, 1, 1), (.82, 0, 0))),
        ]),
        _asset("dock-crane", "Tower crane and cantilever jib", [
            _part("box", (-.24, -.05, 0), (.13, .89, .15), yellow),
            _part("box", (.09, .38, 0), (.80, .11, .16), yellow),
            _part("box", (.10, .45, 0), (.76, .015, .09), steel),
            _part("box", (-.24, -.37, .087), (.10, .02, .015), steel, repeat=((1, 7, 1), (0, .12, 0))),
            _part("box", (-.24, .25, .09), (.21, .18, .16), (59, 88, 101)),
            _part("box", (.34, .05, 0), (.008, .64, .008), dark),
            _part("box", (.34, -.28, 0), (.18, .035, .16), steel),
            _part("box", (-.24, -.48, 0), (.38, .04, .38), dark),
        ]),
        _asset("dock-catwalk", "Open steel catwalk with handrails", [
            _part("box", (0, -.34, 0), (1, .11, .73), steel),
            _part("box", (-.45, -.27, -.37), (.035, .10, .035), yellow, repeat=((6, 1, 2), (.18, 0, .74))),
            _part("box", (0, -.07, -.37), (1, .035, .035), yellow, repeat=((1, 1, 2), (0, 0, .74))),
            _part("box", (-.45, -.32, 0), (.025, .012, .70), (162, 171, 166), repeat=((6, 1, 1), (.18, 0, 0))),
        ]),
        _asset("dock-platform", "Loading platform with safety edge", [
            _part("box", (0, -.33, 0), (1, .34, 1), (76, 88, 91)),
            _part("box", (0, -.145, 0), (.97, .03, .96), (133, 145, 143)),
            _part("box", (0, -.12, .46), (.96, .02, .08), yellow),
            _part("box", (-.41, -.479, -.41), (.12, .038, .12), dark, repeat=((2, 1, 2), (.82, 0, .82))),
        ]),
        _asset("dock-stairs", "Steel access stairs", [
            _part("box", (0, -.36, -.32), (.9, .08, .24), steel),
            _part("box", (0, -.22, -.16), (.9, .08, .24), steel),
            _part("box", (0, -.08, 0), (.9, .08, .24), steel),
            _part("box", (0, .06, .16), (.9, .08, .24), steel),
            _part("box", (0, .20, .32), (.9, .08, .24), steel),
            _part("box", (-.46, -.11, 0), (.025, .68, .84), yellow, repeat=((2, 1, 1), (.92, 0, 0))),
        ]),
        _asset("dock-warehouse", "Pier warehouse facade with lit doors", [
            _part("box", (0, -.03, 0), (.95, .90, .82), (93, 105, 108)),
            _part("box", (0, .45, 0), (1, .07, .91), steel),
            _part("box", (-.31, -.28, .42), (.24, .35, .025), dark, repeat=((3, 1, 1), (.31, 0, 0))),
            _part("box", (-.31, -.09, .44), (.24, .025, .025), yellow, repeat=((3, 1, 1), (.31, 0, 0))),
            _part("box", (-.31, .20, .423), (.18, .14, .025), (117, 169, 177), repeat=((3, 1, 1), (.31, 0, 0))),
            _part("box", (-.31, .29, .45), (.25, .02, .07), dark, repeat=((3, 1, 1), (.31, 0, 0))),
            _part("box", (-.48, 0, .44), (.018, .83, .06), steel, repeat=((2, 1, 1), (.96, 0, 0))),
        ]),
        _asset("dock-floodlight", "Harbor floodlight mast", [
            _part("cylinder", (0, -.02, 0), (.07, .85, .07), steel),
            _part("box", (0, .41, 0), (.62, .045, .08), steel),
            _part("box", (-.24, .35, .055), (.18, .12, .09), (231, 222, 180), repeat=((3, 1, 1), (.24, 0, 0))),
            _part("box", (0, -.46, 0), (.31, .07, .31), dark),
        ]),
        _asset("dock-bollard", "Mooring bollard", [
            _part("cylinder", (0, -.15, 0), (.22, .65, .22), dark),
            _part("cylinder", (0, .19, 0), (.38, .12, .38), steel),
            _part("box", (0, .21, 0), (.46, .08, .12), steel),
            _part("box", (0, -.48, 0), (.40, .04, .4), yellow),
        ]),
    ]


def make_dockyard(seed=0):
    """Harbor chase set with stacked routes, two gantries and a crane silhouette."""
    rng = random.Random(seed)
    scene = _Set("dock")
    scene.add("Concrete apron", "dock-ground", (-1.0, -.055, -11), (29, .11, 38))
    scene.add("Quay edge", "dock-quay", (13.2, -.025, -11), (2.2, .16, 38))
    scene.add("Harbor basin", "dock-water", (21.4, -.115, -11), (14, .22, 38))
    # Clear central run, with stacked containers defining two usable side routes.
    for side in (-1, 1):
        for row, z in enumerate((2.0, -3.4, -8.8, -14.2, -19.6)):
            x = side * (6.6 + (row % 2) * .25)
            scene.add("Ground container", "dock-container-blue" if (row + side) % 2 else "dock-container",
                      (x, 1.35, z), (4.8, 2.7, 2.8), yaw=90 if side < 0 else -90)
            if row in (1, 2, 4):
                scene.add("Stacked container", "dock-container" if row % 2 else "dock-container-blue",
                          (x + side * .16, 4.05, z), (4.8, 2.7, 2.8), yaw=90 if side < 0 else -90)
            if row in (2, 4):
                scene.add("High container", "dock-container-blue" if side < 0 else "dock-container",
                          (x + side * .16, 6.75, z), (4.8, 2.7, 2.8), yaw=90 if side < 0 else -90)
    for z in (-6.0, -18.0):
        scene.add("Overhead container gantry", "dock-gantry", (0, 4.55, z), (14.4, 9.1, 3.1))
        scene.add("Gantry catwalk", "dock-catwalk", (0, 7.8, z), (10.6, 1.1, 1.7))
    scene.add("Quayside tower crane", "dock-crane", (12.3, 9.4, -17.2), (14.0, 18.8, 3.1), yaw=-35)
    for z in (-2.5, -11.7, -23.5):
        scene.add("Loading platform", "dock-platform", (10.15, .75, z), (4.1, 1.5, 3.0))
    scene.add("Access stairs", "dock-stairs", (9.1, .8, -5.2), (1.6, 1.6, 2.2), yaw=90)
    scene.add("Access stairs", "dock-stairs", (-9.5, 1.5, -13.3), (1.6, 3.0, 2.2), yaw=-90)
    for z in (-3.0, -13.0, -23.0):
        scene.add("Quay bollard", "dock-bollard", (12.45, .52, z), (.7, 1.04, .7))
    for x, z in ((-11, 3), (10.5, 3), (-11, -26), (10.5, -26)):
        scene.add("Floodlight mast", "dock-floodlight", (x, 3.5, z), (.75, 7.0, .75))
    for x in (-9.5, -4.4, .7, 5.8):
        scene.add("Pier warehouse", "dock-warehouse", (x, 3.4, -29.8-rng.uniform(0, .15)),
                  (4.9, 6.8+rng.choice((0, .3, .6)), 3.1))
    control = make_object("console", 0)
    control.update(id="dock-crane-controls", name="Crane controls",
                   position=[10.5, .575, -17.2], color=[89, 112, 113])
    scene.objects.append(control)
    return {"version": 3, "name": "Industrial harbor chase", "assets": _dock_assets(),
            "objects": scene.objects, "effects": [], "lighting": "sunset" if seed % 3 == 2 else "neutral",
            "camera": {"position": [28, 23, 31], "look_at": [0, 6.0, -12]}}


def _temple_assets():
    stone = (151, 147, 119)
    shade = (104, 111, 95)
    moss = (67, 111, 74)
    gold = (206, 168, 88)
    return [
        _asset("temple-earth", "Forest floor", [
            _part("box", (0, -.01, 0), (1, .98, 1), (72, 111, 75)),
            _part("box", (-.37, .49, -.34), (.18, .012, .20), (96, 125, 73), repeat=((5, 1, 4), (.18, 0, .22))),
        ]),
        _asset("temple-chasm", "Deep shaded ravine", [
            _part("box", (0, -.20, 0), (1, .60, 1), (34, 65, 68)),
            _part("box", (0, .11, -.44), (.99, .08, .1), (83, 90, 77), repeat=((1, 1, 2), (0, 0, .88))),
        ]),
        _asset("temple-terrace", "Carved stone terrace", [
            _part("box", (0, -.15, 0), (.98, .7, .98), stone),
            _part("box", (0, .21, 0), (.97, .04, .97), shade),
            _part("box", (-.36, .245, -.36), (.22, .02, .22), (194, 184, 145),
                  repeat=((4, 1, 4), (.24, 0, .24))),
            _part("box", (0, .14, .465), (.99, .09, .07), shade),
        ]),
        _asset("temple-stairs", "Broad ceremonial stone steps", [
            _part("box", (0, -.40, .36), (1, .20, .25), stone),
            _part("box", (0, -.23, .20), (1, .20, .25), stone),
            _part("box", (0, -.06, .04), (1, .20, .25), stone),
            _part("box", (0, .11, -.12), (1, .20, .25), stone),
            _part("box", (0, .28, -.28), (1, .20, .25), stone),
            _part("box", (-.46, -.03, .03), (.025, .84, .90), shade, repeat=((2, 1, 1), (.92, 0, 0))),
            _part("box", (-.46, .20, -.44), (.05, .10, .08), moss, repeat=((2, 1, 1), (.92, 0, 0))),
        ]),
        _asset("temple-arch", "Carved gateway and lintel", [
            _part("box", (-.37, -.08, 0), (.22, .84, .43), stone, repeat=((2, 1, 1), (.74, 0, 0))),
            _part("box", (0, .39, 0), (1, .22, .52), stone),
            _part("box", (0, .46, .28), (.86, .06, .05), gold),
            _part("box", (-.37, -.08, .23), (.16, .62, .035), shade, repeat=((2, 1, 1), (.74, 0, 0))),
            _part("box", (-.37, -.37, .235), (.24, .06, .06), moss, repeat=((2, 1, 1), (.74, 0, 0))),
            _part("box", (-.30, .34, .29), (.09, .065, .035), shade, repeat=((7, 1, 1), (.10, 0, 0))),
        ]),
        _asset("temple-tower", "Layered ruin tower with balcony", [
            _part("box", (0, -.17, 0), (.69, .65, .69), stone),
            _part("box", (0, .17, 0), (.86, .075, .86), shade),
            _part("box", (0, .30, 0), (.64, .22, .64), stone),
            _part("box", (0, .43, 0), (.88, .08, .88), (180, 173, 133)),
            _part("box", (-.31, -.15, .36), (.12, .47, .03), shade, repeat=((3, 1, 1), (.31, 0, 0))),
            _part("box", (-.30, .25, .37), (.08, .09, .04), gold, repeat=((3, 1, 1), (.30, 0, 0))),
            _part("box", (-.38, .01, -.38), (.08, .80, .08), moss, repeat=((2, 1, 2), (.76, 0, .76))),
        ]),
        _asset("temple-column", "Carved freestanding column", [
            _part("cylinder", (0, -.03, 0), (.36, .81, .36), stone),
            _part("box", (0, -.45, 0), (.62, .09, .62), shade),
            _part("box", (0, .43, 0), (.67, .11, .67), (188, 177, 139)),
            _part("box", (-.12, -.05, .19), (.035, .72, .018), shade, repeat=((3, 1, 1), (.12, 0, 0))),
            _part("box", (0, .34, .22), (.42, .04, .03), gold),
        ]),
        _asset("temple-bridge", "Suspended timber bridge", [
            _part("box", (0, -.32, 0), (.81, .10, .98), (107, 79, 52)),
            _part("box", (0, -.26, -.45), (.71, .035, .075), (155, 118, 73), repeat=((1, 1, 8), (0, 0, .128))),
            _part("box", (-.43, -.05, -.44), (.035, .44, .04), (128, 97, 64), repeat=((2, 1, 5), (.86, 0, .22))),
            _part("box", (-.43, .15, 0), (.025, .025, .96), (185, 157, 102), repeat=((2, 1, 1), (.86, 0, 0))),
            _part("box", (-.43, -.34, 0), (.08, .07, .96), shade, repeat=((2, 1, 1), (.86, 0, 0))),
        ]),
        _asset("temple-roof", "Stepped shrine roof", [
            _part("box", (0, -.35, 0), (1, .25, 1), stone),
            _part("box", (0, -.13, 0), (.80, .20, .81), shade),
            _part("box", (0, .08, 0), (.62, .20, .63), stone),
            _part("box", (0, .29, 0), (.42, .19, .44), shade),
            _part("box", (0, .43, 0), (.22, .12, .23), gold),
        ]),
        _asset("temple-tree", "Layered jungle canopy tree", [
            _part("cylinder", (0, -.20, 0), (.13, .60, .13), (91, 70, 48)),
            _part("sphere", (0, .24, 0), (.82, .50, .80), (50, 112, 65)),
            _part("sphere", (-.25, .04, .16), (.49, .38, .49), (64, 136, 73)),
            _part("sphere", (.27, .15, -.15), (.45, .36, .45), (39, 101, 63)),
            _part("box", (-.26, -.18, .17), (.025, .54, .025), (88, 119, 57)),
        ]),
        _asset("temple-vines", "Hanging vines on fallen beam", [
            _part("box", (0, .37, 0), (1, .15, .19), shade),
            _part("cylinder", (-.42, -.05, .02), (.035, .71, .035), moss, repeat=((7, 1, 1), (.14, 0, 0))),
            _part("sphere", (-.42, -.36, .02), (.14, .15, .14), (45, 109, 65), repeat=((7, 1, 1), (.14, 0, 0))),
        ]),
        _asset("temple-rock", "Mossy broken stone", [
            _part("box", (0, -.10, 0), (.72, .57, .68), shade, rotation=(0, 16, 8)),
            _part("box", (0, .24, -.04), (.75, .11, .72), stone, rotation=(0, 16, 8)),
            _part("box", (-.24, .30, -.20), (.22, .04, .19), moss),
        ]),
        _asset("temple-brazier", "Ceremonial stone brazier", [
            _part("cylinder", (0, -.22, 0), (.22, .51, .22), shade),
            _part("cylinder", (0, .17, 0), (.62, .24, .62), stone),
            _part("cylinder", (0, .30, 0), (.48, .10, .48), gold),
            _part("cone", (0, .40, 0), (.25, .18, .25), (246, 176, 75)),
        ]),
    ]


def make_temple(seed=0):
    """Jungle ruin with a readable terrace ascent and bridge over a ravine."""
    rng = random.Random(seed)
    scene = _Set("temple")
    scene.add("Near jungle ground", "temple-earth", (0, -.07, 2.4), (31, .14, 24))
    scene.add("Far jungle ground", "temple-earth", (0, -.07, -25.6), (31, .14, 13))
    scene.add("Deep ravine", "temple-chasm", (0, -2.35, -14.0), (31, 4.3, 7.0))
    # Keep paving above the forest floor to prevent coplanar surface flicker.
    scene.add("Starting court", "temple-terrace", (0, -.18, 0), (10.0, .4, 8.0))
    scene.add("Raised approach", "temple-terrace", (0, 1.05, -6.2), (8.0, 2.1, 5.0))
    scene.add("Ceremonial stairs", "temple-stairs", (0, 1.05, -3.7), (4.0, 2.1, 3.4))
    scene.add("Near bridge landing", "temple-terrace", (0, 1.5, -9.3), (5.2, 3.0, 2.8))
    # The plank top is at raw y=-.2425. Its rendered y is about -.257 after
    # the mesh-bound normalization, so this center meets both 3 m landings.
    scene.add("Suspended ravine bridge", "temple-bridge", (0, 3.7706, -14.0), (2.5, 3.0, 8.8))
    scene.add("Far bridge landing", "temple-terrace", (0, 1.5, -18.7), (5.2, 3.0, 2.8))
    scene.add("Far ceremonial stairs", "temple-stairs", (0, 4.3, -20.5), (4.0, 2.6, 3.4))
    scene.add("High temple dais", "temple-terrace", (0, 2.8, -24.0), (12, 5.6, 7.0))
    scene.add("Temple threshold", "temple-arch", (0, 6.8, -22.4), (5.4, 8.0, 2.1))
    scene.add("Temple crown", "temple-roof", (0, 11.1, -26.8), (10.0, 4.8, 6.4))
    for side in (-1, 1):
        scene.add("Side terrace", "temple-terrace", (side*8.0, .72, -5.6), (5.8, 1.45, 10.0))
        scene.add("Side stone stair", "temple-stairs", (side*5.2, .55, -2.3), (2.2, 1.1, 3.0), yaw=side*90)
        scene.add("Ravine sentinel tower", "temple-tower", (side*8.8, 5.0, -10.6), (5.0, 10.0, 5.0))
        scene.add("Temple flank tower", "temple-tower", (side*9.4, 6.2, -25.2), (4.8, 12.4, 4.8))
        scene.add("Gate column", "temple-column", (side*3.35, 3.8, -8.5), (1.2, 7.6, 1.2))
        scene.add("Shrine column", "temple-column", (side*4.3, 5.8, -22.8), (1.4, 6.0, 1.4))
        scene.add("Hanging temple vines", "temple-vines", (side*6.8, 5.6, -23.2), (3.6, 4.8, .5))
        scene.add("Ceremonial brazier", "temple-brazier", (side*3.1, 3.5, -23.1), (.9, 1.4, .9))
    for x, z in ((-13, 4), (-12, -2), (-13, -8), (12.8, 3), (13, -3),
                 (12.5, -8), (-13, -20), (13, -20), (-12.8, -28), (12.8, -28)):
        scene.add("Jungle canopy tree", "temple-tree", (x+rng.uniform(-.25, .25), 3.9, z),
                  (3.6, 7.8+rng.choice((-.3, 0, .3)), 3.6))
    for x, z in ((-5.2, 4), (5.3, 4.3), (-11, -15), (11, -15), (-6.9, -27.2), (6.9, -28)):
        scene.add("Mossy rubble", "temple-rock", (x, .55, z), (1.5, 1.1, 1.4))
    gate = make_object("door", 0)
    gate.update(id="temple-shrine-door", name="Shrine door", position=[0, 7.1, -22.38],
                size=[2.5, 3.0, .2], color=[112, 91, 65])
    scene.objects.append(gate)
    return {"version": 3, "name": "Jungle temple crossing", "assets": _temple_assets(),
            "objects": scene.objects, "effects": [], "lighting": "warm",
            "camera": {"position": [13, 10, 16], "look_at": [0, 4.1, -13]}}


def suggested_targets(scene):
    """Return editable target hints on physical surfaces of the supplied set.

    Points use the renderer's actual mesh-bound normalization. Consumers can
    apply the object's size, yaw and position when converting to world space.
    """
    from asset_geometry import compile_asset

    targets = []
    assets = {asset["id"]: asset for asset in scene.get("assets", [])}
    bounds = {}

    def local(asset_id, raw):
        if asset_id not in bounds:
            vertices, _, _ = compile_asset(assets[asset_id])
            bounds[asset_id] = (vertices.min(axis=0), vertices.max(axis=0))
        lower, upper = bounds[asset_id]
        return [max(-.5, min(.5, float((raw[i] - (lower[i] + upper[i]) / 2) /
                                          (upper[i] - lower[i])))) for i in range(3)]

    specs = {
        "dock-container": [("landing", [0, .4975, 0], "Container roof"),
                           ("climb", [-.47, 0, .495], "Door-side corner frame")],
        "dock-container-blue": [("landing", [0, .4975, 0], "Container roof"),
                                ("climb", [-.47, 0, .495], "Door-side corner frame")],
        "dock-gantry": [("swing_anchor", [0, .365, 0], "Gantry crossbeam underside")],
        "dock-crane": [("swing_anchor", [.34, .325, 0], "Crane jib underside")],
        "dock-catwalk": [("landing", [0, -.285, 0], "Catwalk deck")],
        "dock-platform": [("landing", [0, -.13, 0], "Loading deck")],
        "dock-bollard": [("vault", [0, .25, 0], "Bollard top")],
        "temple-terrace": [("landing", [.12, .255, .12], "Stone paving")],
        "temple-stairs": [("landing", [0, .38, -.28], "Upper step")],
        "temple-bridge": [("landing", [0, -.2425, .062], "Bridge planks")],
        "temple-arch": [("swing_anchor", [0, .28, 0], "Gateway lintel underside")],
        "temple-tower": [("landing", [0, .47, 0], "Tower crown"),
                         ("climb", [0, -.10, .345], "Carved tower face")],
        "temple-column": [("vault", [0, .485, 0], "Column capital")],
        "temple-rock": [("vault", [0, .295, -.04], "Rock ledge")],
    }
    for obj in scene.get("objects", []):
        for kind, local_position, name in specs.get(obj.get("asset"), ()):
            targets.append({"id": f"{obj['id']}-{kind}-{len(targets)}", "name": name,
                            "object_id": obj["id"], "kind": kind,
                            "local_position": local(obj["asset"], local_position)})
    return targets
