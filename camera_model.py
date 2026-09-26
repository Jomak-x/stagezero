"""Validated world-space cameras and per-take hard cuts, independent of the viewer."""
import math
from numbers import Real


MAX_CAMERAS = 64
MAX_CAMERA_POSITION = 10_000.0
MAX_CAMERA_CUTS = 15_000


def _identifier(value, label):
    if not isinstance(value, str) or not 1 <= len(value) <= 80 or not value.strip():
        raise ValueError(f'Invalid {label}')
    return value


def _scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must contain finite numbers')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{label} must contain finite numbers') from None
    if not math.isfinite(result):
        raise ValueError(f'{label} must contain finite numbers')
    return result


def _vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{label} must have {length} components')
    return [_scalar(component, label) for component in value]


def validate_camera(camera):
    """Return a detached, normalized camera record or raise ValueError."""
    if not isinstance(camera, dict) or set(camera) != {'id', 'name', 'position', 'wxyz', 'fov'}:
        raise ValueError('Invalid camera record')
    camera_id = _identifier(camera['id'], 'camera identifier')
    name = _identifier(camera['name'], 'camera name').strip()
    position = _vector(camera['position'], 3, 'Camera position')
    if any(abs(component) > MAX_CAMERA_POSITION for component in position):
        raise ValueError('Camera position is outside the supported world bounds')
    wxyz = _vector(camera['wxyz'], 4, 'Camera orientation')
    norm = math.hypot(*wxyz)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('Camera orientation must be a nonzero quaternion')
    fov = _scalar(camera['fov'], 'Camera field of view')
    if not 0 < fov < math.pi:
        raise ValueError('Camera field of view must be between 0 and pi radians')
    return dict(id=camera_id, name=name, position=position,
                wxyz=[component / norm for component in wxyz], fov=fov)


def validate_cameras(cameras):
    """Validate a project's shared camera library without retaining input aliases."""
    if not isinstance(cameras, list) or len(cameras) > MAX_CAMERAS:
        raise ValueError(f'A project supports at most {MAX_CAMERAS} cameras')
    validated = [validate_camera(camera) for camera in cameras]
    if len({camera['id'] for camera in validated}) != len(validated):
        raise ValueError('Duplicate camera identifiers')
    return validated


def validate_camera_cuts(cuts, length, camera_ids=None):
    """Validate an ordered hard-cut timeline, including optional camera references."""
    if not isinstance(cuts, list) or len(cuts) > MAX_CAMERA_CUTS:
        raise ValueError('Invalid camera cut timeline')
    validated, identifiers = [], set()
    previous = -1
    for cut in cuts:
        if not isinstance(cut, dict) or set(cut) != {'id', 'frame', 'camera_id'}:
            raise ValueError('Invalid camera cut')
        cut_id = _identifier(cut['id'], 'camera cut identifier')
        camera_id = _identifier(cut['camera_id'], 'camera reference')
        frame = cut['frame']
        if type(frame) is not int or not 0 <= frame < length or frame <= previous:
            raise ValueError('Camera cut frames must be unique, ordered and within the take')
        if not validated and frame != 0:
            raise ValueError('The first camera cut must begin at frame 0')
        if cut_id in identifiers:
            raise ValueError('Duplicate camera cut identifiers')
        if camera_ids is not None and camera_id not in camera_ids:
            raise ValueError('Camera cut references an unknown camera')
        validated.append(dict(id=cut_id, frame=frame, camera_id=camera_id))
        identifiers.add(cut_id)
        previous = frame
    return validated


def copy_camera_cuts(cuts, length):
    """Keep absolute cut frames that remain inside an edited take's new length."""
    return [dict(cut) for cut in cuts if cut['frame'] < length]


def camera_at_frame(cuts, frame):
    """Return the last hard cut at or before frame; no interpolation or looping."""
    camera_id = None
    for cut in cuts:
        if cut['frame'] > frame:
            break
        camera_id = cut['camera_id']
    return camera_id
