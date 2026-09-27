"""Narrow completion checks for finite G1 actions; never synthesize poses.

These checks reject obvious truncation, not guarantee stylistic or physical
accuracy. Unrecognized continuous movements retain their planned duration.
"""
import re
import numpy as np
from motion_quality import ROOT, SHOULDERS, JOINT_INDEX
from story_recovery import NEGATION, is_recovery_motion, recovery_completion_frame, _upright_frames

MAX_ACTION_FRAMES = 750
MAX_ACTION_EXTENSIONS = 7


def action_spec(prompt, previous_prompt=''):
    text = prompt.lower()
    if NEGATION.search(text):
        return None
    if re.search(r'\bback[ -]?flips?\b', text):
        return {'kind': 'backflip'}
    if re.search(r'\b(?:fall(?:s|ing)?|collaps(?:e|es|ing))\b', text):
        return {'kind': 'fall'}
    if is_recovery_motion(text):
        return {'kind': 'recovery'}
    if re.search(r'\b(?:stop(?:s|ping)?|halts?|comes? to (?:a stop|rest)|stand(?:s|ing)? still)\b', text):
        return {'kind': 'stop'}
    if re.search(r'\bstand(?:s|ing)? (?:upright|straight|up)\b', text):
        return {'kind': 'stand'}
    distance = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)\b', text)
    if distance and re.search(r'\b(?:walk\w*|run\w*|jog\w*|sprint\w*|travel\w*)\b', text):
        return {'kind': 'travel', 'meters': float(distance.group(1))}
    return None


def action_prompt(prompt, previous_prompt=''):
    """Describe the necessary transition instead of repeating a static pose."""
    if re.fullmatch(r'A person danc(?:es|ing)(?: in place)?[.!]?', prompt.strip(), re.I):
        return 'A person dances energetically with rhythmic footwork and swinging arms.'
    spec = action_spec(prompt, previous_prompt)
    if spec and spec['kind'] in ('stand', 'recovery') and re.search(r'\b(?:squat\w*|crouch\w*|kneel\w*)\b', previous_prompt, re.I):
        return 'A person straightens both legs and stands upright.'
    if spec and spec['kind'] == 'stop':
        return 'A person stops and stands still.'
    return prompt


def _run_end(flags, length=10, minimum=1):
    run = 0
    for end, flag in enumerate(flags, 1):
        run = run + 1 if flag else 0
        if run >= length and end >= minimum:
            return end
    return None


def _leg_length(p):
    lengths=[]
    for side in ('left', 'right'):
        hip,knee,ankle=(p[:, JOINT_INDEX[f'{side}_{name}_skel']] for name in ('hip_yaw','knee','ankle_roll'))
        lengths.append(np.linalg.norm(hip-knee,axis=1)+np.linalg.norm(knee-ankle,axis=1))
    return np.mean(lengths,axis=0)


def _fallen(p):
    legs = _leg_length(p)
    ankles = p[:, [JOINT_INDEX['left_ankle_roll_skel'],JOINT_INDEX['right_ankle_roll_skel']],1].mean(axis=1)
    torso = p[:,SHOULDERS].mean(axis=1)-p[:,ROOT]
    return ((legs > 1e-5) & ((p[:,ROOT,1]-ankles) < .48*legs)
            & (torso[:,1] < .55*np.linalg.norm(torso,axis=1)))


def action_completion_frame(spec, positions, prior_positions=None):
    p=np.asarray(positions)
    if p.ndim != 3 or p.shape[1:] != (34,3) or not len(p) or not np.isfinite(p).all():
        return None
    kind=spec['kind']
    if kind == 'recovery':
        return recovery_completion_frame(p)
    if kind == 'stand':
        return _run_end(_upright_frames(p), minimum=25)
    if kind == 'fall':
        return _run_end(_fallen(p), minimum=15)
    if kind == 'travel':
        origin=p[0,ROOT,[0,2]] if prior_positions is None or not len(prior_positions) else np.asarray(prior_positions)[-1,ROOT,[0,2]]
        distance=np.linalg.norm(p[:,ROOT][:,[0,2]]-origin,axis=1)
        hits=np.flatnonzero(distance >= spec['meters'])
        return int(hits[0])+1 if len(hits) else None
    if kind == 'stop':
        speed=np.r_[np.inf,np.linalg.norm(np.diff(p[:,ROOT][:,[0,2]],axis=0),axis=1)*25]
        upright=_upright_frames(p)
        # Assess a 0.4 s settling window; one early deceleration sample
        # must not reject an otherwise stopped ending.
        for end in range(15,len(p)+1):
            if speed[end-10:end].mean() < .20 and speed[end-1] < .20 and upright[end-10:end].all():
                return end
        return None
    if kind == 'backflip':
        torso=p[:,SHOULDERS].mean(axis=1)-p[:,ROOT]
        inverted=torso[:,1] < -.35*np.linalg.norm(torso,axis=1)
        inversion=_run_end(inverted,length=3)
        if inversion is None:
            return None
        # Require a rise as well as inversion, then a sustained upright ending.
        base=p[0,ROOT,1] if prior_positions is None or not len(prior_positions) else np.asarray(prior_positions)[-1,ROOT,1]
        if p[:inversion,ROOT,1].max() < base + .10*float(np.median(_leg_length(p))):
            return None
        landing=_run_end(_upright_frames(p[inversion:]),minimum=10)
        return inversion+landing if landing is not None else None
    return None


def action_finished(spec, positions, prior_positions=None):
    p=np.asarray(positions)
    endpoint=action_completion_frame(spec,p,prior_positions)
    if endpoint is None:
        return False
    if spec['kind'] in ('recovery','stand','backflip'):
        return len(p)>=10 and bool(_upright_frames(p[-10:]).all())
    if spec['kind'] == 'fall':
        return len(p)>=10 and bool(_fallen(p[-10:]).all())
    if spec['kind'] == 'stop':
        speed=np.linalg.norm(np.diff(p[-11:,ROOT][:,[0,2]],axis=0),axis=1)*25
        return len(p)>=11 and float(speed.mean()) < .20 and float(speed[-1]) < .20 and bool(_upright_frames(p[-10:]).all())
    return True


def action_motion_target(spec, positions, prior_positions=None):
    """Use the pod's native waypoint condition for explicitly measured travel.

    The first accepted travel chunk establishes direction. Every subsequent
    waypoint stays within the backend's 3 m per-horizon limit. Output poses
    are still entirely generated and quality-checked by the model.
    """
    if spec is None or spec['kind'] != 'travel' or positions is None or len(positions) < 10:
        return None
    p=np.asarray(positions)
    origin=p[0,ROOT,[0,2]] if prior_positions is None or not len(prior_positions) else np.asarray(prior_positions)[-1,ROOT,[0,2]]
    first_end=min(104,len(p))-1
    direction=p[first_end,ROOT,[0,2]]-origin
    distance=float(np.linalg.norm(direction))
    if distance < .1:
        return None
    destination=origin+direction/distance*spec['meters']
    current=p[-1,ROOT,[0,2]]
    delta=destination-current
    remaining=float(np.linalg.norm(delta))
    if remaining < .01:
        return None
    return {'position_xz':(current+delta/remaining*min(2.7,remaining)).tolist(),'frame':51}


def generate_action_chunk(backend, request_id, prompt, history, spec=None,
                          positions=None, prior_positions=None):
    target=action_motion_target(spec,positions,prior_positions)
    if target is None and history is not None:
        target=directional_motion_target(prompt,positions,prior_positions)
    options={}
    if target is not None and getattr(backend,'supports_motion_target',False):
        options['motion_target']=target
    if spec is not None and getattr(backend,'supports_generation_options',False):
        if spec['kind'] == 'backflip':
            options['generation_options']={'profile':'responsive','candidates':3}
        elif spec['kind'] == 'fall' and active_action_lead_in(prior_positions):
            options['generation_options']={'profile':'expressive','candidates':3}
    return backend.generate(request_id,prompt,history,**options)


def active_action_lead_in(positions):
    """Strong conditioning helps a fall interrupt an ongoing articulated action.

    Settled lead-ins keep the ordinary profile. Inspect only the last second,
    so a preceding run does not force a new profile after a completed stop.
    """
    if positions is None or len(positions) < 2:
        return False
    p=np.asarray(positions)[-25:]
    scale=float(np.median(_leg_length(p)))
    if scale < 1e-5:
        return False
    local=p-p[:,ROOT,None,:]
    speed=np.linalg.norm(np.diff(local,axis=0),axis=2).mean(axis=1)*25/scale
    return float(speed.mean()) >= .30


def action_attempt_limit(spec):
    # Difficult finite stunts get a bounded extra sampling budget; rejected
    # candidates never alter the accepted history or visible take.
    return 5 if spec is not None and spec['kind']=='backflip' else 3


def action_sample_suitable(prompt, positions):
    """Reject an obvious freeze during a generic continuous dance request.

    This is a motion-energy check, not a dance-style classifier. Deliberately
    slow styles, poses, and short transition fragments are not gated here.
    """
    if not re.search(r'\bdances? (?:energetically|rhythmically)\b',prompt,re.I):
        return True
    p=np.asarray(positions)
    if len(p)<25:
        return True
    scale=float(np.median(_leg_length(p)))
    if scale<1e-5:
        return False
    local=p-p[:,ROOT,None,:]
    speed=np.linalg.norm(np.diff(local,axis=0),axis=2).mean(axis=1)*25/scale
    return float(np.mean(speed >= .30)) >= .50 and float(speed.mean()) >= .30


def directional_motion_target(prompt, positions, prior_positions=None):
    """Condition explicit sidesteps/backward walking in the initial body frame."""
    if NEGATION.search(prompt):
        return None
    sideways=bool(re.search(r'\b(?:sidestep\w*|side[- ]steps?)\b',prompt,re.I))
    backward=bool(re.search(r'\bwalk\w*\b',prompt,re.I) and re.search(r'\bbackward(?:s)?\b',prompt,re.I))
    if not (sideways or backward):
        return None
    initial=np.asarray(prior_positions)[-1] if prior_positions is not None and len(prior_positions) else None
    if initial is None and positions is not None and len(positions):
        # A scene's first action has no earlier beat. Its first accepted pose
        # still establishes a body frame for subsequent conditioned chunks.
        initial=np.asarray(positions)[0]
    if initial is None:
        return None
    current=np.asarray(positions)[-1] if positions is not None and len(positions) else initial
    if sideways:
        left=bool(re.search(r'\bleft\b',prompt,re.I));right=bool(re.search(r'\bright\b',prompt,re.I))
        if left == right:
            return None
        axis=(initial[JOINT_INDEX['left_hip_yaw_skel']]-initial[JOINT_INDEX['right_hip_yaw_skel']])[[0,2]]
        sign=1 if left else -1
    else:
        axis=sum((initial[JOINT_INDEX[f'{side}_toe_base']]-initial[JOINT_INDEX[f'{side}_ankle_roll_skel']])[[0,2]] for side in ('left','right'))
        sign=-1
    norm=float(np.linalg.norm(axis))
    if norm<1e-5:
        return None
    delta=axis/norm*(1.2*sign)
    return {'position_xz':(current[ROOT,[0,2]]+delta).tolist(),'frame':51}
