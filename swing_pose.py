"""Fixed-length Core27 swing/carry poses, with actual Core motion residuals.

The flight path and contact IK are authored constraints, not model predictions.
Core samples add bounded body articulation without weakening paired contacts.
Coordinates are Y-up metres, +Z forward; snapshot root is the hero's pelvis.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motion_bridge import _layout, _swing, _two_bone


def _smooth(x):
    x = float(np.clip(x, 0, 1))
    return x*x*(3-2*x)


def _blend_rotation(a, b, amount):
    return a @ Rotation.from_rotvec(Rotation.from_matrix(a.T @ b).as_rotvec()*amount).as_matrix()


def _yaw(angle):
    return Rotation.from_euler('y', angle).as_matrix()


class PoseController:
    """Per-scene controller. set_clip atomically adopts genuine Core output.

    get_pose returns numpy arrays; JSON callers should convert with .tolist().
    Model clips may be [T,27,...] or [actors,T,27,...]. They crossfade over .5s
    on adoption and loop with a short seam blend. Root locomotion is excluded.
    """
    def __init__(self, clip_path=None, mj_scale=.94):
        self.names, self.parents, self.neutral = _layout()
        self.idx = {n:i for i,n in enumerate(self.names)}
        self.offsets = self.neutral-self.neutral[np.maximum(self.parents, 0)]
        self.scales = np.array([1., float(mj_scale)])
        self._clip = None
        self._old_clip = None
        self._old_sample = None
        self._last_model_sample = None
        self._adopt_time = None
        self._lock = threading.Lock()
        self._last_time = 0.
        if clip_path is not None:
            with np.load(Path(clip_path), allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}
                self.set_clip(data['positions'], data['rotations'], metadata)

    def set_clip(self, positions, rotations, source_metadata=None):
        """Thread-safe, inexpensive adoption; never moves the scene root."""
        p, r = np.asarray(positions,dtype=float), np.asarray(rotations,dtype=float)
        if p.ndim == 3:
            p, r = p[None], r[None]
        if p.ndim != 4 or p.shape[2:] != (27,3) or r.shape != p.shape[:-1]+(3,3):
            raise ValueError('Core clip must be [actors,T,27,3] with matching global rotations')
        if not np.isfinite(p).all() or not np.isfinite(r).all() or p.shape[1] < 2:
            raise ValueError('Core clip needs at least two finite frames')
        if np.max(np.abs(r.swapaxes(-1,-2)@r-np.eye(3))) > .01 or np.min(np.linalg.det(r)) < .99:
            raise ValueError('Core global rotations must be proper SO(3) rotations')
        meta = dict(source_metadata or {})
        fps = float(meta.get('fps',20))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Core fps must be positive')
        # Anatomical motion relative to the generated root yaw; world translation
        # and generated global heading never enter live physics.
        relative = np.einsum('atji,atbik->atbjk',r[:,:,0],r)
        local_p = np.einsum('atji,atbi->atbj',r[:,:,0],p-p[:,:,:1])
        with self._lock:
            self._old_clip = self._clip
            # A refresh can arrive before the previous .5s blend has finished.
            # Preserve the pose actually emitted, rather than jumping to that
            # previous clip's unblended target on the first frame of a refresh.
            self._old_sample = self._last_model_sample
            self._clip = (local_p.copy(),relative.copy(),fps,meta)
            self._adopt_time = None

    def _sample(self, clip, t):
        if clip is None:
            return None
        p,r,fps,meta = clip
        frames = p.shape[1]
        # Ping-pong avoids endpoint discontinuities without introducing root drift.
        phase = (t*fps) % (2*(frames-1))
        frame = min(phase,2*(frames-1)-phase)
        i,j = int(frame), min(int(frame)+1,frames-1)
        alpha = frame-i
        pp=(1-alpha)*p[:,i]+alpha*p[:,j]
        rr=np.empty_like(r[:,i])
        for a in range(len(rr)):
            for b in range(27):
                rr[a,b]=_blend_rotation(r[a,i,b],r[a,j,b],alpha)
        return pp,rr

    def _model(self,t):
        with self._lock:
            clip, old = self._clip, self._old_clip
            old_sample = self._old_sample
            if self._adopt_time is None:
                self._adopt_time=t
            age=t-self._adopt_time
        now=self._sample(clip,t)
        if now is None:
            return None, {'source':'authored constrained pose; no Core clip adopted','model_weight':0.}
        mix=_smooth(age/.5)
        previous=old_sample if old_sample is not None else self._sample(old,t)
        p,r=now
        prior_p=np.broadcast_to(self.neutral,p.shape) if previous is None else previous[0][np.minimum(np.arange(len(p)),len(previous[0])-1)]
        p=prior_p*(1-mix)+p*mix
        for a in range(len(r)):
            for j in range(27):
                prior=np.eye(3) if previous is None else previous[1][min(a,len(previous[1])-1),j]
                r[a,j]=_blend_rotation(prior,r[a,j],mix)
        with self._lock:
            if self._clip is clip:
                self._last_model_sample=(p.copy(),r.copy())
        provenance=dict(clip[3])
        provenance.update({'source':'native Core motion + authored flight/contact IK',
                           'model_weight':.16,'adoption_blend':mix,
                           'model_weights':{'spine_rotation':.10,'hero_foot_residual':.16},
                           'constraints':'fixed bones; paired hand IK; authored flight and head aim'})
        return (p,r), provenance

    def _body(self, root, body_rotation, scale, model, actor, t, flight, carry, kiss):
        p=np.empty((27,3)); r=np.empty((27,3,3))
        p[0]=root; r[0]=body_rotation
        off=self.offsets*scale
        for j in range(1,27):
            par=self.parents[j]
            p[j]=p[par]+r[par]@off[j]
            r[j]=r[par]
            if j in (1,2,3,4) and model is not None:
                # Relative global deviations stay bounded so generated walking
                # poses cannot collapse the authored aerial silhouette.
                source=model[1][min(actor,len(model[1])-1),j]
                delta=Rotation.from_matrix(source).as_rotvec()
                delta=np.clip(delta,-.5,.5)*.10*(1-kiss)
                r[j]=body_rotation@Rotation.from_rotvec(delta).as_matrix()
        return p,r

    def _limb(self,p,r,side,kind,target,pole,scale):
        a,b,c = (('Arm','ForeArm','Hand') if kind=='arm' else ('UpLeg','Leg','Foot'))
        a,b,c=[self.idx[side+n] for n in (a,b,c)]
        off=self.offsets*scale
        mid,end=_two_bone(p[a],target,pole,np.linalg.norm(off[b]),np.linalg.norm(off[c]),r[a]@off[b])
        r[a]=_swing(r[a]@off[b],mid-p[a])@r[a]
        p[b]=mid
        r[b]=_swing(r[a]@off[c],end-mid)@r[a]
        p[c]=end; r[c]=r[b]
        for j in range(c+1,27):
            if self.parents[j]==c:
                p[j]=p[c]+r[c]@off[j];r[j]=r[c]

    def get_pose(self,snapshot,t):
        s=snapshot if isinstance(snapshot,dict) else vars(snapshot)
        root=np.asarray(s.get('root',[0,.96,0]),dtype=float)
        yaw=float(s.get('yaw',0.))
        carry=float(np.clip(s.get('carry_amount',s.get('carryamount',0.)),0,1))
        kiss=float(np.clip(s.get('kiss_amount',s.get('kissamount',0.)),0,1))
        phase=str(s.get('phase','idle'))
        flight=float(np.clip(s.get('flight_amount',s.get('airborne_amount',1. if phase in ('swing','swinging','airborne','launch') else 0.)),0,1))
        model,provenance=self._model(float(t))
        rotation=_yaw(yaw)
        velocity=rotation.T@np.asarray(s.get('velocity',[0,0,0]),dtype=float)
        lean=flight*np.clip(velocity[2]*.012,-.12,.12)+kiss*.13
        bank=flight*np.clip(-velocity[0]*.008,-.1,.1)
        body=rotation@Rotation.from_euler('xz',[lean,bank]).as_matrix()
        hp,hr=self._body(root,body,1.,model,0,t,flight,carry,kiss)
        # During a grounded ending MJ steps around to the front before leaning in.
        ending=phase in ('landing','settling','landed','dismount','kiss','kissing','finished') or kiss>0
        stand_local=np.array([.46, -.96*(1-self.scales[1]),.05])
        if ending:
            stand_local=np.array([.035,-.96*(1-self.scales[1]),.42])
        pickup=s.get('mj_pickup',s.get('MJpickup',s.get('mj_root')))
        if pickup is not None and not ending:
            standing=np.asarray(pickup,dtype=float)
        else:
            standing=root+rotation@stand_local
        carried=root+body@np.array([0,.22,-.29])
        mj_root=standing*(1-carry)+carried*carry
        if 'mj_root' in s:
            mj_root=np.asarray(s['mj_root'],dtype=float)
        mj_yaw=yaw+np.pi*(1-carry) if ending else yaw
        mj_yaw=float(s.get('mj_yaw',mj_yaw))
        mj_body=_yaw(mj_yaw)@Rotation.from_euler('x',.15*carry+kiss*.08).as_matrix()
        mp,mr=self._body(mj_root,mj_body,self.scales[1],model,1,t,flight,carry,kiss)
        # Feet hang back in flight, then land under the hips; two-bone IK keeps
        # limb lengths exact through every interpolation.
        for side,sgn in [('Right',-1),('Left',1)]:
            goal=np.array([sgn*(.13+.06*flight),-.87+.30*flight,-.25*flight])
            if model is not None:
                j=self.idx[side+'Foot']
                source=model[0][0,j]
                goal+=np.clip(source-self.neutral[j],-.25,.25)*.16*(1-kiss)
            self._limb(hp,hr,side,'leg',root+body@goal,root+body@np.array([sgn*.17,-.3,.5]),1.)
            stand_foot=mj_root+_yaw(mj_yaw)@np.array([sgn*.13,-.88*self.scales[1],0])
            carry_foot=root+body@np.array([sgn*.32,-.28,-.13])
            target=stand_foot*(1-carry)+carry_foot*carry
            self._limb(mp,mr,side,'leg',target,mj_root+body@np.array([sgn*.42,-.25,.6]),self.scales[1])
            for p,r,scale,heading in [(hp,hr,1.,yaw),(mp,mr,self.scales[1],mj_yaw)]:
                foot,toe=self.idx[side+'Foot'],self.idx[side+'ToeBase']
                r[foot]=_yaw(heading)
                p[toe]=p[foot]+r[foot]@self.offsets[toe]*scale
                r[toe]=r[foot]
        # Hero right palm cups the passenger thigh; left hand holds the web.
        thigh=(mp[self.idx['RightUpLeg']]*.35+mp[self.idx['RightLeg']]*.65)+body@np.array([-.015,-.025,0])
        right_rest=root+body@np.array([-.3,.03,.08])
        right_target=(1-carry)*right_rest+carry*thigh
        if kiss>0:
            right_target=(1-kiss)*right_target+kiss*(mp[self.idx['LeftArm']]+rotation@np.array([.025,-.13,0]))
        self._limb(hp,hr,'Right','arm',right_target,root+body@np.array([-.60,.25,-.25]),1.)
        web_target=root+body@np.array([.22,.98,.10])
        left_rest=root+body@np.array([.3,.03,.08])
        left_target=(1-flight)*left_rest+flight*web_target
        if kiss>0:
            left_target=(1-kiss)*left_target+kiss*(mp[self.idx['RightArm']]+rotation@np.array([-.025,-.13,0]))
        self._limb(hp,hr,'Left','arm',left_target,root+body@np.array([.55,.7,-.18]),1.)
        shoulder_targets=[]
        for side,sgn in [('Right',-1),('Left',1)]:
            target=hp[self.idx[side+'Arm']]+body@np.array([0,.015,.025])
            shoulder_targets.append(target)
            rest=mj_root+mj_body@np.array([sgn*.27,.04,.10])
            target=rest*(1-carry)+target*carry
            if kiss>0:
                opposite='Left' if side=='Right' else 'Right'
                near=hp[self.idx[opposite+'Arm']]+rotation@np.array([0,-.04,.03])
                target=target*(1-kiss)+near*kiss
            self._limb(mp,mr,side,'arm',target,mj_root+body@np.array([sgn*.52,.31,.1]),self.scales[1])
        # Head rotation affects the skinned face; neck chain already leans
        # naturally. Leave head joint centres apart, faces meet in front.
        for p,r,heading,other in [(hp,hr,yaw,mp),(mp,mr,mj_yaw,hp)]:
            head=self.idx['Head']
            forward=_yaw(heading)@np.array([0,0,1.])
            toward=other[head]-p[head]
            toward/=max(np.linalg.norm(toward),1e-8)
            base=_yaw(heading)
            r[head]=_blend_rotation(base,_swing(forward,toward)@base,kiss)
        grips={'mj_right_shoulder_m':float(np.linalg.norm(mp[self.idx['RightHand']]-shoulder_targets[0])),
               'mj_left_shoulder_m':float(np.linalg.norm(mp[self.idx['LeftHand']]-shoulder_targets[1])),
               'hero_thigh_m':float(np.linalg.norm(hp[self.idx['RightHand']]-thigh)),
               'active':carry>.98}
        positions=np.stack([hp,mp]); rotations=np.stack([hr,mr])
        heads=positions[:,self.idx['Head']]
        forward=rotations[:,self.idx['Head']]@np.array([0.,0.,1.])
        # CoreSkin's head joint sits behind its face. The bound mesh reaches
        # z=.151 in head-local coordinates; use the rendered mouth location,
        # including each actor's scale, rather than a generic head-radius proxy.
        mouth_local=np.array([0.,.015,.15])*self.scales[:,None]
        face_centres=heads+np.einsum('aij,aj->ai',rotations[:,self.idx['Head']],mouth_local)
        faces={'kiss_amount':kiss,'head_forward':forward,'face_centres':face_centres,
               'mouth_local':[0.,.015,.15],
               'head_distance_m':float(np.linalg.norm(heads[0]-heads[1]))}
        return {'positions':positions,'rotations':rotations,'actor_scales':self.scales.copy(),
                'grip_metrics':grips,'face_factors':faces,'provenance':provenance,
                'web_hand':hp[self.idx['LeftHand']].copy()}


_default_controller=None


def get_pose(snapshot,t):
    global _default_controller
    if _default_controller is None:
        _default_controller=PoseController()
    return _default_controller.get_pose(snapshot,t)
