"""Tailored plum suit body for the Patron character (metres, +Z up, -Y front)."""

import math

from common import M, curve_tube, ellipsoid, mesh_object


TAU = 2.0 * math.pi


# The jacket is a continuous fitted shell.  The forward shift and increased
# front-to-back radius below the chest give the character his heavyset profile.
JACKET_PROFILE = [
    # z, half width, centre y, half depth
    (0.862, 0.307, 0.014, 0.266),
    (0.871, 0.315, 0.014, 0.272),
    (0.905, 0.319, 0.011, 0.280),
    (0.960, 0.324, 0.008, 0.300),
    (1.025, 0.324, 0.008, 0.314),
    (1.090, 0.317, 0.009, 0.309),
    (1.160, 0.301, 0.011, 0.277),
    (1.230, 0.298, 0.014, 0.232),
    (1.305, 0.305, 0.018, 0.195),
    (1.370, 0.321, 0.023, 0.171),
    (1.405, 0.335, 0.024, 0.159),
    (1.432, 0.331, 0.025, 0.149),
    (1.446, 0.294, 0.025, 0.139),
    (1.460, 0.190, 0.020, 0.100),
    (1.470, 0.112, 0.000, 0.068),
    (1.475, 0.075, -0.003, 0.052),
    (1.482, 0.074, -0.003, 0.051),
]


def _profile(z):
    if z <= JACKET_PROFILE[0][0]:
        return JACKET_PROFILE[0][1:]
    if z >= JACKET_PROFILE[-1][0]:
        return JACKET_PROFILE[-1][1:]
    for lo, hi in zip(JACKET_PROFILE, JACKET_PROFILE[1:]):
        if lo[0] <= z <= hi[0]:
            t = (z - lo[0]) / (hi[0] - lo[0])
            return tuple(a + (b - a) * t for a, b in zip(lo[1:], hi[1:]))
    raise AssertionError("Unreachable jacket profile height")


def _front_y(x, z, outset=0.0):
    rx, cy, ry = _profile(z)
    fraction = min(abs(x) / rx, 0.995)
    return cy - ry * math.sqrt(1.0 - fraction * fraction) - outset


def _rings_object(name, rings, material, segments=48, subdiv=1):
    """Join bottom-to-top elliptical rings with outward-facing quad strips."""
    verts = []
    faces = []
    for ring in rings:
        z, rx, cy, ry, xcentre, hem_drop = ring
        for j in range(segments):
            theta = TAU * j / segments
            x = xcentre + rx * math.cos(theta)
            y = cy + ry * math.sin(theta)
            # The front quarters hang a touch lower than the side vents.
            zz = z - hem_drop * max(0.0, -math.sin(theta)) ** 2
            verts.append((x, y, zz))
    for i in range(len(rings) - 1):
        a0 = i * segments
        b0 = (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((a0 + j, a0 + k, b0 + k, b0 + j))
    return mesh_object(name, verts, faces, material, subsurf=subdiv)


def _surface_panel(name, rows, side, material, offset=0.012, across=4):
    """Cloth strip shaped by (z, inner x, outer x) rows, from bottom up."""
    verts = []
    faces = []
    for z, inner, outer in rows:
        for j in range(across + 1):
            x = side * (inner + (outer - inner) * j / across)
            verts.append((x, _front_y(x, z, offset), z))
    width = across + 1
    for i in range(len(rows) - 1):
        for j in range(across):
            a = i * width + j
            b = (i + 1) * width + j
            faces.append((a, a + 1, b + 1, b))
    return mesh_object(name, verts, faces, material, subsurf=0)


def _jacket_shell():
    rings = []
    for z, rx, cy, ry in JACKET_PROFILE:
        drop = 0.008 if z < 0.872 else (0.004 if z < 0.906 else 0.0)
        rings.append((z, rx, cy, ry, 0.0, drop))
    _rings_object("Jacket - fitted continuous shell", rings, M["suit"], subdiv=1)

def _shirt_and_lapels():
    # Black shirt occupies the deep V visible between the folded jacket fronts.
    shirt_rows = [
        (1.182, 0.000, 0.011),
        (1.225, 0.000, 0.039),
        (1.285, 0.000, 0.071),
        (1.345, 0.000, 0.102),
        (1.410, 0.000, 0.116),
        (1.456, 0.000, 0.110),
        (1.467, 0.000, 0.091),
        (1.474, 0.000, 0.065),
    ]
    for side, label in ((-1, "left"), (1, "right")):
        _surface_panel(
            "Black shirt front - " + label,
            shirt_rows,
            side,
            M["shirt"],
            offset=0.004,
            across=5,
        )

    # A small exposed upper chest triangle makes the open collar legible.
    skin_rows = [
        # z, half width; the lower rows follow the shirt surface, while the
        # last two rise into and then tuck under the sculpted neck.
        (1.376, 0.003),
        (1.423, 0.027),
        (1.461, 0.041),
        (1.467, 0.041),
        (1.475, 0.040),
        (1.482, 0.038),
    ]
    for side, label in ((-1, "left"), (1, "right")):
        skin_vertices = []
        skin_faces = []
        for z, width in skin_rows:
            for j in range(4):
                t = j / 3
                x = side * width * t
                if z <= 1.467:
                    y = _front_y(x, z, 0.022)
                elif z <= 1.475:
                    y = -0.064 + 0.004 * t
                else:
                    y = -0.053 + 0.004 * t
                skin_vertices.append((x, y, z))
        for i in range(len(skin_rows) - 1):
            for j in range(3):
                a = i * 4 + j
                skin_faces.append((a, a + 1, a + 5, a + 4))
        mesh_object("Open collar skin - " + label, skin_vertices, skin_faces, M["skin"], subsurf=0)

    # Jacket lapels have their own surface.  The outer edge changes direction
    # at the notch, as in the reference, without floating in front of the body.
    lapel_rows = [
        (1.186, 0.013, 0.036),
        (1.225, 0.040, 0.072),
        (1.275, 0.069, 0.130),
        (1.325, 0.094, 0.180),
        (1.370, 0.117, 0.220),
        (1.390, 0.125, 0.226),
        (1.405, 0.127, 0.198),
        (1.417, 0.127, 0.170),
        (1.451, 0.111, 0.151),
        (1.465, 0.092, 0.113),
        (1.470, 0.081, 0.101),
    ]
    for side, label in ((-1, "left"), (1, "right")):
        _surface_panel(
            "Jacket notched lapel - " + label,
            lapel_rows,
            side,
            M["suit"],
            offset=0.008,
            across=5,
        )
        edge = [
            (side * outer, _front_y(side * outer, z, 0.009), z)
            for z, inner, outer in lapel_rows
        ]
        curve_tube("Lapel edge seam - " + label, edge, 0.0015, M["seam"])

        # The dark turned shirt collar sits just inside the jacket's lapel.
        collar_rows = [
            (1.393, 0.038, 0.050),
            (1.431, 0.041, 0.103),
            (1.462, 0.045, 0.091),
            (1.471, 0.066, 0.083),
            (1.480, 0.071, 0.080),
        ]
        _surface_panel(
            "Open black shirt collar - " + label,
            collar_rows,
            side,
            M["shirt"],
            offset=0.014,
            across=3,
        )

    # Two understated shirt buttons are visible within the opening.
    for index, z in enumerate((1.355, 1.292)):
        y = _front_y(0.0, z, 0.007)
        ellipsoid(
            "Open shirt button %d" % (index + 1),
            (0.0, y, z),
            (0.0051, 0.0024, 0.0051),
            M["button"],
            segments=16,
            rings=8,
        )


def _coat_details():
    # A narrow centre break starts below the single-breasted closure.
    _surface_panel(
        "Jacket lower front lining gap",
        [
            (0.858, 0.0, 0.032),
            (0.890, 0.0, 0.028),
            (0.940, 0.0, 0.018),
            (0.990, 0.0, 0.006),
        ],
        1,
        M["lining"],
        offset=0.006,
        across=3,
    )
    _surface_panel(
        "Jacket lower front lining gap mirror",
        [
            (0.858, 0.0, 0.032),
            (0.890, 0.0, 0.028),
            (0.940, 0.0, 0.018),
            (0.990, 0.0, 0.006),
        ],
        -1,
        M["lining"],
        offset=0.006,
        across=3,
    )
    # Closure follows the belly's actual front curvature.
    for index, (x, z) in enumerate(((0.008, 1.147), (-0.016, 1.034))):
        ellipsoid(
            "Plum jacket front button %d" % (index + 1),
            (x, _front_y(x, z, 0.004), z),
            (0.0108, 0.0032, 0.0108),
            M["button"],
            segments=20,
            rings=10,
        )
        # Short horizontal keyhole-like stitch alongside each button.
        curve_tube(
            "Jacket buttonhole %d" % (index + 1),
            [
                (x + 0.019, _front_y(x + 0.019, z, 0.005), z),
                (x + 0.041, _front_y(x + 0.041, z, 0.005), z),
            ],
            0.0012,
            M["seam"],
        )

    for side, label in ((-1, "left"), (1, "right")):
        # Shallow waist pockets follow the rounded jacket body.
        pocket_rows = [(0.985, 0.176, 0.266), (1.025, 0.174, 0.264)]
        _surface_panel(
            "Jacket waist pocket flap - " + label,
            pocket_rows,
            side,
            M["suit"],
            offset=0.005,
            across=6,
        )
        curve_tube(
            "Jacket waist pocket lip - " + label,
            [
                (side * x, _front_y(side * x, 1.027, 0.006), 1.027)
                for x in (0.174, 0.194, 0.219, 0.244, 0.264)
            ],
            0.0015,
            M["seam"],
        )
        # Darts fall gently toward the waist, creating tailored side panels.
        curve_tube(
            "Jacket front dart - " + label,
            [
                (side * x, _front_y(side * x, z, 0.005), z)
                for x, z in ((0.243, 1.280), (0.254, 1.210), (0.264, 1.125), (0.274, 1.015), (0.270, 0.915))
            ],
            0.0009,
            M["seam"],
        )
        # A faint shoulder/armhole line starts high where the sleeve is inset.
        curve_tube(
            "Jacket shoulder seam - " + label,
            [
                (side * x, y, z)
                for x, y, z in (
                    (0.198, -0.050, 1.463),
                    (0.259, -0.046, 1.451),
                    (0.315, -0.033, 1.432),
                    (0.347, -0.009, 1.410),
                )
            ],
            0.0011,
            M["seam"],
        )

    # One breast welt, as in the front reference view.
    _surface_panel(
        "Jacket breast welt",
        [(1.326, 0.166, 0.232), (1.338, 0.166, 0.232)],
        1,
            M["suit"],
        offset=0.005,
        across=5,
    )
    curve_tube(
        "Breast pocket upper stitch",
        [
            (x, _front_y(x, 1.342, 0.006), 1.342)
            for x in (0.166, 0.182, 0.199, 0.216, 0.233)
        ],
        0.0012,
        M["seam"],
    )


def _sleeve(side, label):
    # A curved arm centreline avoids the straight-cylinder silhouette.
    rows = [
        ((0.240, 0.014, 1.394), 0.025, 0.036),
        ((0.265, 0.010, 1.401), 0.055, 0.068),
        ((0.301, 0.006, 1.401), 0.083, 0.095),
        ((0.337, 0.003, 1.392), 0.108, 0.109),
        ((0.380, 0.002, 1.357), 0.103, 0.105),
        ((0.427, 0.001, 1.296), 0.095, 0.095),
        ((0.490, -0.002, 1.213), 0.084, 0.087),
        ((0.537, -0.007, 1.152), 0.077, 0.080),
        ((0.596, -0.014, 1.073), 0.069, 0.069),
        ((0.652, -0.021, 0.996), 0.060, 0.060),
        ((0.675, -0.025, 0.965), 0.057, 0.057),
        ((0.680, -0.025, 0.958), 0.056, 0.056),
    ]
    segments = 24
    verts = []
    faces = []
    for i, (centre, upper_r, depth_r) in enumerate(rows):
        left = rows[max(0, i - 1)][0]
        right = rows[min(len(rows) - 1, i + 1)][0]
        dx = right[0] - left[0]
        dz = right[2] - left[2]
        norm = math.hypot(dx, dz)
        ux, uz = -dz / norm, dx / norm
        for j in range(segments):
            theta = TAU * j / segments
            c, s = math.cos(theta), math.sin(theta)
            x = side * (centre[0] + ux * upper_r * c)
            y = centre[1] + depth_r * s
            # Flatten the top of the section: a suit shoulder slopes away
            # from the neck instead of peaking like a round pipe end.
            z = centre[2] + uz * upper_r * c * 0.43
            verts.append((x, y, z))
    for i in range(len(rows) - 1):
        a0 = i * segments
        b0 = (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((a0 + j, b0 + j, b0 + k, a0 + k))
    # Cap the inner end deep inside the jacket.  Its circumference remains
    # buried as the sleeve swells through the shoulder seam.
    root = len(verts)
    verts.append((side * rows[0][0][0], rows[0][0][1], rows[0][0][2]))
    for j in range(segments):
        faces.append((root, (j + 1) % segments, j))
    mesh_object("Tailored jacket sleeve - " + label, verts, faces, M["suit"], subsurf=1)

    # A thin black shirt cuff is just visible inside the plum sleeve end.
    cuff_verts = []
    cuff_faces = []
    for t, radius in ((0.0, 0.0555), (1.0, 0.0555)):
        cx = side * (0.679 + 0.008 * t)
        cy = -0.025 - 0.001 * t
        cz = 0.960 - 0.009 * t
        for j in range(segments):
            theta = TAU * j / segments
            cuff_verts.append((cx + side * 0.8 * radius * math.cos(theta), cy + radius * math.sin(theta), cz + 0.6 * radius * math.cos(theta)))
    for j in range(segments):
        k = (j + 1) % segments
        cuff_faces.append((j, segments + j, segments + k, k))
    mesh_object("Black shirt cuff - " + label, cuff_verts, cuff_faces, M["shirt"], subsurf=0)

    # A few tiny cuff buttons run along the outer sleeve edge.
    for i, (x, z) in enumerate(((0.610, 1.030), (0.626, 1.010), (0.642, 0.990))):
        ellipsoid(
            "Sleeve cuff button %s %d" % (label, i + 1),
            (side * x, -0.078, z),
            (0.004, 0.002, 0.004),
            M["button"],
            segments=12,
            rings=8,
        )


def _trousers():
    # The top bridge disappears under the jacket but joins the two legs.
    _rings_object(
        "Suit trouser upper seat",
        [
            (0.788, 0.255, 0.022, 0.150, 0.0, 0.0),
            (0.813, 0.275, 0.024, 0.167, 0.0, 0.0),
            (0.872, 0.303, 0.022, 0.211, 0.0, 0.0),
            (0.916, 0.306, 0.018, 0.227, 0.0, 0.0),
        ],
        M["suit"],
        subdiv=1,
    )
    for side, label in ((-1, "left"), (1, "right")):
        leg_rows = [
            # z, width, y centre, depth, absolute x centre
            (0.146, 0.088, 0.025, 0.102, 0.158),
            (0.153, 0.090, 0.025, 0.105, 0.158),
            (0.188, 0.094, 0.028, 0.109, 0.157),
            (0.260, 0.100, 0.030, 0.112, 0.155),
            (0.365, 0.106, 0.027, 0.113, 0.153),
            (0.470, 0.109, 0.022, 0.122, 0.150),
            (0.540, 0.114, 0.023, 0.125, 0.148),
            (0.660, 0.125, 0.023, 0.146, 0.146),
            (0.760, 0.133, 0.025, 0.166, 0.144),
            (0.824, 0.130, 0.025, 0.172, 0.144),
            (0.834, 0.128, 0.025, 0.172, 0.144),
        ]
        rings = [(z, rx, cy, ry, side * cx, 0.0) for z, rx, cy, ry, cx in leg_rows]
        _rings_object("Tailored suit trouser leg - " + label, rings, M["suit"], subdiv=1)

        # Pressed front crease follows each leg's slight taper.
        crease = [
            (side * cx, cy - ry - 0.002, z)
            for z, rx, cy, ry, cx in leg_rows[2:-1]
        ]
        curve_tube("Trouser pressed crease - " + label, crease, 0.0010, M["seam"])
        # Angled cloth breaks above the shoes and a restrained knee wrinkle.
        for idx, (z, lean) in enumerate(((0.190, 0.015), (0.223, -0.011), (0.442, 0.010))):
            cx = 0.158 if z < 0.3 else 0.150
            ry = 0.108 if z < 0.3 else 0.121
            cy = 0.026 if z < 0.3 else 0.022
            curve_tube(
                "Trouser front cloth break %s %d" % (label, idx + 1),
                [
                    (side * (cx + dx), cy - ry * math.sqrt(max(0.0, 1.0 - (dx / 0.10) ** 2)) - 0.002, z + lean * (dx / 0.10))
                    for dx in (-0.071, -0.037, 0.0, 0.037, 0.071)
                ],
                0.0009,
                M["seam"],
            )


def build_body():
    """Build the reference character's static fitted suit and shirt."""
    _jacket_shell()
    _shirt_and_lapels()
    _coat_details()
    _sleeve(-1, "left")
    _sleeve(1, "right")
    _trousers()
