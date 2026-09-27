"""Reusable action sets with portable, prop-relative interaction targets."""
from scene_composition import validate_scene


def make_cinematic(name,seed=0):
    if name=='Rooftop swing district':
        from cinematic_city import make_swing_city,suggested_targets
        scene=make_swing_city(seed)
    elif name in ('Harbor chase','Jungle temple'):
        from cinematic_adventure import make_dockyard,make_temple,suggested_targets
        scene=(make_dockyard if name=='Harbor chase' else make_temple)(seed)
    else:raise ValueError('Unknown cinematic scene')
    scene['targets']=suggested_targets(scene)
    return validate_scene(scene)
