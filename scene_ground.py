"""Choose when an authored scene surface replaces the studio's generic floor.

Scene objects use their rendered, normalized bounding-box dimensions, so this
policy applies equally to procedural assets and primitive platforms. It does
not depend on asset names or the internal shape recipe.
"""


def has_authored_ground(objects):
    """Return whether a broad, low horizontal surface supplies the scene floor."""
    for obj in objects:
        if obj.get('kind') not in ('custom', 'platform'):
            continue
        x, height, z = obj['size']
        center_y = obj['position'][1]
        bottom, top = center_y - height / 2, center_y + height / 2
        if (x >= 4.0 and z >= 4.0 and x * z >= 30.0
                and height <= .75 and -.8 <= bottom <= .2
                and -.25 <= top <= .45):
            return True
    return False


def studio_surface_visibility(objects, *, show_grid=True, show_platform=True):
    """Resolve stage visibility while retaining the user's grid/stage choices."""
    show_studio = not has_authored_ground(objects)
    return show_studio, show_studio and show_grid, show_studio and show_platform
