"""Hands and dress shoes for the suited patron character.

Scene units are metres, +Z is up, and the character faces -Y.
"""

import math

from common import M, curve_tube, mesh_object


TAU = math.tau


def _close_cap(vertices, faces, ring, center):
    cap = len(vertices)
    vertices.append(center)
    for i in range(len(ring)):
        faces.append((cap, ring[i], ring[(i + 1) % len(ring)]))


def _loft_tube(vertices, faces, sections, sides=16):
    """Append a closed, tapered organic shape with horizontal section rings.

    A section is (x, y, z, x_radius, y_radius); the ring orientation
    remains consistent when mirroring a hand.
    """
    rings = []
    for x, y, z, rx, ry in sections:
        ring = []
        for i in range(sides):
            angle = TAU * i / sides
            ring.append(len(vertices))
            vertices.append((x + rx * math.cos(angle),
                             y + ry * math.sin(angle), z))
        rings.append(ring)
    for lower, upper in zip(rings, rings[1:]):
        for i in range(sides):
            j = (i + 1) % sides
            faces.append((lower[i], upper[i], upper[j], lower[j]))
    x, y, z, _, _ = sections[0]
    _close_cap(vertices, faces, rings[0], (x, y, z + 0.001))
    x, y, z, _, _ = sections[-1]
    cap = len(vertices)
    vertices.append((x, y, z - 0.001))
    for i in range(sides):
        faces.append((cap, rings[-1][(i + 1) % sides], rings[-1][i]))


def _make_hand(side):
    import bpy

    s = float(side)
    verts, faces = [], []
    palm = [
        (.967, .681, -.024, .039, .024),
        (.952, .685, -.024, .041, .025),
        (.931, .694, -.025, .044, .026),
        (.908, .704, -.026, .049, .027),
        (.885, .713, -.027, .051, .026),
        (.860, .721, -.028, .050, .024),
        (.838, .727, -.028, .046, .021),
        (.819, .729, -.027, .038, .017),
        (.810, .730, -.026, .029, .012),
    ]
    _loft_tube(verts, faces,
               [(s * x, y, z, rx, ry) for z, x, y, rx, ry in palm], 24)

    # The roots overlap the palm before a fine voxel union. Individual
    # fingertips remain free, with a small forward curl toward the viewer.
    digits = [
        (.690, .775, .0112, .0123),  # index
        (.714, .763, .0120, .0130),  # middle
        (.738, .773, .0116, .0126),  # ring
        (.760, .790, .0100, .0112),  # little
    ]
    for x, tip, rx, ry in digits:
        sections = [
            (s*x, -.029, .847, rx*1.12, ry*1.18),
            (s*(x + .001), -.032, .825, rx, ry),
            (s*(x + .002), -.036, max(tip + .026, .801), rx*.90, ry*.94),
            (s*(x + .003), -.042, tip + .012, rx*.83, ry*.84),
            (s*(x + .004), -.046, tip + .004, rx*.60, ry*.62),
            (s*(x + .004), -.047, tip, rx*.24, ry*.27),
        ]
        _loft_tube(verts, faces, sections, 16)

    # Bodyward thumb emerges from the palm edge and lies relaxed beside it.
    thumb = [
        (.684, -.024, .903, .0185, .0170),
        (.673, -.030, .884, .0180, .0170),
        (.659, -.037, .861, .0156, .0156),
        (.647, -.041, .841, .0125, .0130),
        (.641, -.044, .828, .0082, .0088),
        (.640, -.045, .824, .0036, .0040),
    ]
    _loft_tube(verts, faces,
               [(s*x, y, z, rx, ry) for x, y, z, rx, ry in thumb], 16)

    hand = mesh_object(('Right' if side > 0 else 'Left') + '_Hand',
                       verts, faces, M['skin'], subsurf=0)
    # A volumetric union removes the hard intersections at the knuckles
    # while retaining separate tips at this resolution.
    remesh = hand.modifiers.new(name='Organic hand union', type='REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = .0022
    remesh.use_smooth_shade = True
    bpy.context.view_layer.objects.active = hand
    hand.select_set(True)
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    hand.select_set(False)
    for face in hand.data.polygons:
        face.use_smooth = True
    return hand


_FOOTPRINT = [
    (-.247, .001), (-.243, .025), (-.237, .044), (-.228, .057),
    (-.213, .066), (-.204, .070),
    (-.178, .075), (-.135, .078), (-.080, .073), (-.025, .068),
    (.025, .067), (.080, .067), (.118, .060), (.143, .035),
    (.150, .001),
]

_UPPER = [
    (-.244, .018, .072), (-.241, .027, .076),
    (-.236, .038, .082), (-.228, .052, .087),
    (-.216, .060, .092), (-.198, .065, .097),
    (-.165, .068, .101), (-.120, .069, .110),
    (-.065, .065, .119), (-.010, .061, .133),
    (.042, .059, .144), (.089, .057, .142),
    (.125, .048, .126), (.146, .020, .082),
    (.150, .001, .057),
]


def _interpolate(samples, y, column):
    if y <= samples[0][0]:
        return samples[0][column]
    if y >= samples[-1][0]:
        return samples[-1][column]
    def slope(index):
        if index == 0:
            return (samples[1][column]-samples[0][column]) / (samples[1][0]-samples[0][0])
        if index == len(samples)-1:
            return (samples[-1][column]-samples[-2][column]) / (samples[-1][0]-samples[-2][0])
        previous_span = samples[index][0]-samples[index-1][0]
        next_span = samples[index+1][0]-samples[index][0]
        previous = (samples[index][column]-samples[index-1][column])/previous_span
        following = (samples[index+1][column]-samples[index][column])/next_span
        if previous*following <= 0:
            return 0.0
        weight_previous = 2*next_span+previous_span
        weight_following = next_span+2*previous_span
        return (weight_previous+weight_following) / (weight_previous/previous+weight_following/following)

    for index in range(len(samples)-1):
        y0, y1 = samples[index][0], samples[index+1][0]
        if y0 <= y <= y1:
            t = (y-y0)/(y1-y0)
            h = y1-y0
            start = samples[index][column]
            finish = samples[index+1][column]
            return ((2*t**3-3*t*t+1)*start
                    + (t**3-2*t*t+t)*h*slope(index)
                    + (-2*t**3+3*t*t)*finish
                    + (t**3-t*t)*h*slope(index+1))
    return samples[-1][column]


def _shoe_top(y, x_local):
    width = max(_interpolate(_UPPER, y, 1), .001)
    crest = _interpolate(_UPPER, y, 2)
    proportion = min(abs(x_local)/width, .999)
    return .043 + (crest-.043)*(.5+.5*math.sqrt(1-proportion*proportion))


def _make_sole(side):
    center = side*.150
    outline = []
    for y, width in _FOOTPRINT:
        outline.append((width, y))
    for y, width in reversed(_FOOTPRINT[1:-1]):
        outline.append((-width, y))

    verts, faces = [], []
    rings = []
    # Contour changes give the leather edge a deliberate welt and bevel.
    for z, factor in [( .016, .94), (.020, 1.00), (.035, 1.00),
                      (.041, .965), (.045, .920)]:
        ring = []
        for x, y in outline:
            ring.append(len(verts))
            verts.append((center + x*factor, y, z))
        rings.append(ring)
    count = len(outline)
    for a, b in zip(rings, rings[1:]):
        for i in range(count):
            j = (i+1)%count
            faces.append((a[i], a[j], b[j], b[i]))
    # Caps turn the multi-ring welt into one watertight solid.
    bottom = len(verts)
    verts.append((center, -.047, .016))
    top = len(verts)
    verts.append((center, -.047, .045))
    for i in range(count):
        j = (i+1)%count
        faces.extend([(bottom, rings[0][j], rings[0][i]),
                      (top, rings[-1][i], rings[-1][j])])
    mesh_object(('Right' if side > 0 else 'Left') + '_Shoe_Welt',
                verts, faces, M['sole'], subsurf=1)
    curve_tube(('Right' if side > 0 else 'Left') + '_Welt_Stitch',
               [(center+x*.973, y, .040) for x, y in outline],
               .0010, M['sole'], cyclic=True)

    # The heel is a straight, shallow stack under the rear third.
    heel_outline = [(-.004, -.049), (.052, -.062), (.113, -.056),
                    (.144, -.030), (.149, 0), (.144, .030),
                    (.113, .056), (.052, .062), (-.004, .049)]
    heel_outline.reverse()
    v = [(center+x, y, z) for z in (.012, .024)
         for y, x in heel_outline]
    n = len(heel_outline)
    f = []
    for i in range(n):
        j = (i+1)%n
        f.append((i, j, n+j, n+i))
    f.append(tuple(reversed(range(n))))
    f.append(tuple(range(n,2*n)))
    mesh_object(('Right' if side > 0 else 'Left') + '_Heel',
                v, f, M['sole'], subsurf=1)


def _make_upper(side):
    center = side*.150
    verts, faces = [], []
    sections = []
    for i in range(80):
        y = -.244 + (.394*i/79)
        width = _interpolate(_UPPER, y, 1)
        crest = _interpolate(_UPPER, y, 2)
        ring = []
        for k in range(24):
            a = TAU*k/24
            x = center + width*math.cos(a)
            z = .043 + (crest-.043)*(.5+.5*math.sin(a))
            ring.append(len(verts))
            verts.append((x,y,z))
        sections.append(ring)
    for a,b in zip(sections,sections[1:]):
        for k in range(24):
            j = (k+1)%24
            faces.append((a[k],b[k],b[j],a[j]))
    _close_cap(verts,faces,sections[0],(center,-.245,.069))
    cap=len(verts)
    verts.append((center,.151,.055))
    for k in range(24):
        faces.append((cap,sections[-1][(k+1)%24],sections[-1][k]))
    mesh_object(('Right' if side > 0 else 'Left') + '_Shoe_Upper',
                verts,faces,M['leather'],subsurf=1)

    label = 'Right' if side > 0 else 'Left'
    # Wing-tip style toe panel, following the actual top contour.
    toe_y = -.141
    width = _interpolate(_UPPER,toe_y,1)*.966
    toe_line = []
    for k in range(25):
        x = -width + 2*width*k/24
        toe_line.append((center+x,toe_y,_shoe_top(toe_y,x)+.0014))
    curve_tube(label+'_Toe_Cap_Seam',toe_line,.0011,M['sole'])

    # Eye stays and low dark laces follow the domed vamp.
    rows = [-.039,-.014,.011,.036,.061,.085]
    for rail in (-1,1):
        rail_x = rail*.025
        rail_points = [(center+rail_x,y,_shoe_top(y,rail_x)+.0020)
                       for y in [-.052]+rows+[.095]]
        curve_tube(label+('_Inner' if rail<0 else '_Outer')+'_Eye_Stay',
                   rail_points,.0013,M['leather'])
        for row,y in enumerate(rows):
            x = rail*.020
            eyelet=[]
            for k in range(12):
                angle=TAU*k/12
                xx=x+.0029*math.cos(angle)
                yy=y+.0029*math.sin(angle)
                eyelet.append((center+xx,yy,_shoe_top(yy,xx)+.0021))
            curve_tube(f'{label}_Eyelet_{rail}_{row}',eyelet,.00085,
                       M['sole'],cyclic=True)
    for row in range(len(rows)-1):
        y0,y1=rows[row],rows[row+1]
        for direction in (-1,1):
            x0=direction*.020
            x1=-x0
            points=[(center+x0,y0,_shoe_top(y0,x0)+.0022),
                    (center, (y0+y1)/2,_shoe_top((y0+y1)/2,0)+.0032),
                    (center+x1,y1,_shoe_top(y1,x1)+.0022)]
            curve_tube(f'{label}_Lace_{row}_{direction}',points,.00145,
                       M['leather'])

    # Counter stitching curls around the quarter, visible in side view.
    counter=[]
    y=.103
    width=_interpolate(_UPPER,y,1)*.94
    for k in range(19):
        x=-width+2*width*k/18
        counter.append((center+x,y,_shoe_top(y,x)+.0012))
    curve_tube(label+'_Heel_Counter_Seam',counter,.0011,M['sole'])


def build_extremities():
    """Build both relaxed hands and both polished black lace-up shoes."""
    for side in (-1,1):
        _make_hand(side)
        _make_sole(side)
        _make_upper(side)
