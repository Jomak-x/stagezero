"""Hand-shaped stylized head: continuous skin surface, fitted eyes and swept hair."""
from math import sin, cos, sqrt, exp, pi, asin
from common import mesh_object, ellipsoid, curve_tube, M

# Cross sections deliberately hold the mandibular corners and broad temples.
_PROFILE = [
 (1.475,.079,.057,.050), (1.510,.079,.060,.060),
 (1.530,.073,.075,.062), (1.545,.076,.100,.073),
 (1.560,.091,.117,.086), (1.580,.108,.123,.096),
 (1.606,.115,.119,.106), (1.638,.120,.114,.112),
 (1.670,.126,.113,.115), (1.705,.124,.113,.117),
 (1.738,.121,.114,.117), (1.774,.119,.111,.114),
 (1.802,.110,.102,.105), (1.821,.083,.082,.083),
 (1.834,.026,.029,.029), (1.837,.002,.002,.002)]

def _profile(z):
    for i in range(len(_PROFILE)-1):
        a,b = _PROFILE[i:i+2]
        if z <= b[0]:
            u=(z-a[0])/(b[0]-a[0]); u=max(0,min(1,u))
            # Smooth interpolation in each section, with shared cardinal tangents.
            before=_PROFILE[max(0,i-1)]; after=_PROFILE[min(len(_PROFILE)-1,i+2)]
            out=[]
            for j in range(1,4):
                m0=(b[j]-before[j])/(b[0]-before[0])*(b[0]-a[0])
                m1=(after[j]-a[j])/(after[0]-a[0])*(b[0]-a[0])
                out.append((2*u**3-3*u*u+1)*a[j]+(u**3-2*u*u+u)*m0+(-2*u**3+3*u*u)*b[j]+(u**3-u*u)*m1)
            return out
    return _PROFILE[-1][1:]

def _g(x,z,cx,cz,sx,sz):
    return exp(-((x-cx)/sx)**2-((z-cz)/sz)**2)

def _face_delta(x,z):
    d=0.0
    # Frontal planes and cheekbones transition into the temporal sidewall.
    for side in (-1,1):
        d-=.017*_g(x,z,side*.081,1.667,.035,.035)
        d-=.010*_g(x,z,side*.056,1.743,.043,.014)
        d+=.010*_g(x,z,side*.054,1.724,.029,.014)
        d+=.0065*_g(x,z,side*.082,1.638,.022,.026)
        d-=.012*_g(x,z,side*.090,1.589,.024,.034)
        d-=.012*_g(x,z,side*.018,1.656,.012,.010)
        # A softly depressed nasolabial transition, not an applied line.
        d+=.003*_g(x,z,side*.033,1.641,.010,.020)
    # Integrated bridge, cartilage and nostril wings.
    d-=.025*_g(x,z,0,1.697,.014,.040)
    d-=.040*_g(x,z,0,1.664,.020,.016)
    d-=.010*_g(x,z,0,1.682,.016,.028)
    d-=.009*_g(x,z,0,1.745,.016,.018)
    # Firm muzzle, philtrum and broad chin with a shallow centre cleft.
    d-=.010*_g(x,z,0,1.622,.043,.026)
    d-=.014*_g(x,z,0,1.575,.048,.020)
    d+=.0025*_g(x,z,0,1.576,.006,.010)
    d+=.003*_g(x,z,0,1.641,.004,.010)
    # Frown sculpting, sculpted as relief in the forehead itself.
    d+=.0026*_g(x,z,-.009,1.767,.0023,.018)
    d+=.0026*_g(x,z,.009,1.767,.0023,.018)
    return d

def _front_y(x,z):
    w,front,back=_profile(z)
    r=min(.99999,abs(x)/w)
    y=-front*(1-r*r)**.28
    return y+_face_delta(x,z)

def _head_surface():
    vs=[]; fs=[]; n=192; rows=157
    for j in range(rows):
        z=1.475+(1.837-1.475)*j/(rows-1)
        w,front,back=_profile(z)
        for k in range(n):
            t=2*pi*k/n; x=w*sin(t); c=cos(t)
            if c>=0:
                y=-front*c**.56+_face_delta(x,z)*c**1.5
            else:
                y=-back*c
            vs.append((x,y,z))
    for j in range(rows-1):
        for k in range(n):
            a=j*n+k; b=j*n+(k+1)%n
            fs.append((a,b,b+n,a+n))
    fs.append(tuple(reversed(range(n))))
    fs.append(tuple((rows-1)*n+k for k in range(n)))
    obj=mesh_object('Head | sculpted jaw cheeks nose forehead',vs,fs,M['skin'],subsurf=1)
    # Subtle, smooth stubble coloration is exported as ordinary vertex colors.
    colors=obj.data.color_attributes.new(name='SkinTone',type='FLOAT_COLOR',domain='POINT')
    base=tuple(M['skin'].diffuse_color)
    cool=(.21,.185,.169)
    def smooth(a,b,v):
        t=max(0,min(1,(v-a)/(b-a)))
        return t*t*(3-2*t)
    for i,(x,y,z) in enumerate(vs):
        front=smooth(.025,-.060,y)
        lower=smooth(1.540,1.566,z)*(1-smooth(1.622,1.668,z))
        strength=.40*lower*front
        colors.data[i].color=tuple(base[j]*(1-strength)+cool[j]*strength for j in range(3))+(1,)
    shade=M['skin'].copy();shade.name='Skin | subtle jaw stubble vertex color'
    node=shade.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='SkinTone'
    shade.node_tree.links.new(node.outputs['Color'],shade.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    obj.data.materials[0]=shade
    return obj

def _eye(side):
    cx=side*.0525; cz=1.7255; width=.0210
    # Fitted almond surfaces create a realistic small aperture, no projecting globes.
    vs=[]; fs=[]; steps=48; rows=12
    for j in range(rows+1):
        v=j/rows
        for i in range(steps+1):
            q=-1+2*i/steps
            arch=max(0,1-q*q)**.62
            x=cx+q*width
            slope=side*q*.0015
            lo=cz-.0036*arch+slope
            hi=cz+.0044*arch+slope
            z=lo+(hi-lo)*v
            # Front of the ocular surface bows only 3 mm out of the aperture.
            y=-.1090-.0030*arch*sin(pi*v)
            vs.append((x,y,z))
    for j in range(rows):
        for i in range(steps):
            a=j*(steps+1)+i
            fs.append((a,a+1,a+steps+2,a+steps+1))
    mesh_object(('L' if side<0 else 'R')+' eye | small almond sclera',vs,fs,M['eye_white'],subsurf=1)
    # Dark irises are shallow in depth and partly covered by the upper eyelid.
    ellipsoid(('L' if side<0 else 'R')+' iris',(cx,-.1125,cz+.0002),(.0050,.0007,.0045),M['iris'],segments=40,rings=24)
    ellipsoid(('L' if side<0 else 'R')+' pupil',(cx,-.1132,cz+.0002),(.0025,.00035,.0029),M['pupil'],segments=32,rings=20)
    ellipsoid(('L' if side<0 else 'R')+' eye glint',(cx-.0012,-.1136,cz+.0015),(.00065,.00025,.00065),M['eye_white'],segments=20,rings=12)
    # Skin bands join the eye aperture to the continuous underlying brow/cheek.
    for upper in (True,False):
        lv=[]; lf=[]
        for r in range(5):
            v=r/4
            for i in range(steps+1):
                q=-1+2*i/steps; arch=max(0,1-q*q)**.62
                x=cx+q*(width+.003*v)
                slope=side*q*.0015
                z=cz+((.0044 if upper else -.0036)*arch)+slope
                z += (1 if upper else -1)*v*(.0030+.004*arch)
                y0=-.1100-.0007*arch
                yf=_front_y(x,z)-.0005
                y=y0*(1-v)+yf*v
                lv.append((x,y,z))
        for r in range(4):
            for i in range(steps):
                a=r*(steps+1)+i; lf.append((a,a+1,a+steps+2,a+steps+1))
        mesh_object(('L' if side<0 else 'R')+(' upper lid' if upper else ' lower lid'),lv,lf,M['skin'],subsurf=1)
    # Thin lash line tucked into the upper lid.
    pts=[]
    for i in range(33):
        q=-.98+1.96*i/32; a=max(0,1-q*q)**.62
        pts.append((cx+q*width,-.1110-.0007*a,cz+.0044*a+side*q*.0015))
    curve_tube(('L' if side<0 else 'R')+' upper eyelid crease',pts,.00055,M['skin_shadow'])
    # Geometric tapered brow follows the angry inward/downward line.
    points=[(.022,1.732,.0034),(.034,1.736,.0058),(.054,1.742,.0064),(.078,1.747,.0058),(.093,1.741,.0008)]
    bv=[]; bf=[]
    for xx,zz,hh in points:
        x=side*xx
        for zadd,depth in [(-hh,.0008),(0,.0026),(hh,.0008)]:
            z=zz+zadd; bv.append((x,_front_y(x,z)-depth,z))
    for i in range(len(points)-1):
        for j in range(2):
            a=i*3+j; bf.append((a,a+3,a+4,a+1))
    mesh_object(('L' if side<0 else 'R')+' thick angled brow',bv,bf,M['hair'],subsurf=1)

def _lips():
    # Closed, slightly downturned mouth, defined by restrained sculpted vermilion.
    steps=60
    def seam(q):
        return 1.6215-.0050*abs(q)**1.65+.0006*exp(-((abs(q)-.25)/.20)**2)
    for upper in (True,False):
        vs=[];fs=[]
        for r in range(7):
            v=r/6
            for i in range(steps+1):
                q=-1+2*i/steps; x=.043*q; taper=(1-q*q)**.7
                z=seam(q)+(1 if upper else -1)*(.0023 if upper else .0033)*taper*v
                y=_front_y(x,z)-.0006-.0009*sin(pi*v)*taper
                vs.append((x,y,z))
        for r in range(6):
            for i in range(steps):
                a=r*(steps+1)+i;fs.append((a,a+1,a+steps+2,a+steps+1))
        mesh_object('Upper lip' if upper else 'Lower lip',vs,fs,M['lips'],subsurf=1)
    pts=[]
    for i in range(49):
        q=-.98+1.96*i/48;x=.043*q;z=seam(q)
        pts.append((x,_front_y(x,z)-.0012,z))
    curve_tube('Closed mouth line',pts,.00065,M['skin_shadow'])
    # Nostrils sit against the lower wings of the nose rather than drilled holes.
    for s in (-1,1):
        pts=[]
        for i in range(17):
            u=i/16*pi
            x=s*(.015+.005*cos(u));z=1.653-.0018*sin(u)
            pts.append((x,_front_y(x,z)-.0008,z))
        curve_tube(('L' if s<0 else 'R')+' nostril shadow',pts,.00125,M['skin_shadow'])

def _ears():
    for s in (-1,1):
        vs=[];fs=[];n=64;radial=18
        for r in range(radial+1):
            u=r/radial
            for i in range(n):
                t=2*pi*i/n
                x=s*(.129+.018*u*sin(t))
                z=1.685+.038*u*cos(t)
                y=-.013+.011*(1-u*u)-.009*exp(-((u-.81)/.18)**2)
                vs.append((x,y,z))
        for r in range(radial):
            for i in range(n):
                a=r*n+i;b=r*n+(i+1)%n;fs.append((a,b,b+n,a+n))
        back=len(vs);vs.append((s*.129,.010,1.685))
        for i in range(n):fs.append((radial*n+i,radial*n+(i+1)%n,back))
        mesh_object(('L' if s<0 else 'R')+' ear | sculpted pinna',vs,fs,M['skin'],subsurf=1)
        pts=[]
        for i in range(29):
            t=-.5+3.9*i/28
            pts.append((s*(.130+.008*sin(t)),-.014,1.686+.025*cos(t)))
        curve_tube(('L' if s<0 else 'R')+' ear antihelix',pts,.0022,M['skin'])
        ellipsoid(('L' if s<0 else 'R')+' ear canal',(s*.131,-.014,1.684),(.004,.001,.0055),M['skin_shadow'],segments=24,rings=16)

def _hair():
    # Hair is a connected fitted cap with low sculpted ridges, not strand tubes.
    vs=[];fs=[];n=192;rows=64
    for j in range(rows):
        u=j/(rows-1)
        for k in range(n):
            t=2*pi*k/n;c=cos(t)
            # Central hairline and temple recession, with a short clean back.
            if c>=0:
                zb=1.700+.088*min(1,c/.50)**.55-.005*exp(-(sin(t)/.17)**2)
            else:
                zb=1.698-.088*(-c)**.8
            phi0=asin(max(-.95,min(.95,(zb-1.716)/.166)))
            phi=phi0+(pi/2-.008-phi0)*u
            twist=.20*sin(pi*u)
            tt=t+twist
            x=.133*cos(phi)*sin(tt)
            ry=.128 if cos(tt)>=0 else .125+.019*(1-u)**2
            cc=cos(tt)
            y=(-ry*cos(phi)*cc**.56 if cc>=0 else -ry*cos(phi)*cc)+.005*sin(pi*u)-.004*u
            z=1.716+.166*sin(phi)+.003*cos(t)*sin(pi*u)
            # Raised front sweep and flatter rear crown break the helmet silhouette.
            z-=.009*u
            z+=.020*exp(-((y+.044)/.034)**2)*sqrt(u)*exp(-((x+.012)/.095)**2)
            # Grooves follow the backward comb direction across the entire crown.
            # Irregular phase and sub-millimetre relief avoid repeated radial ribs.
            phase=173*x+1.9*sin(16*y)+1.2*sin(21*x+5*y)
            detail=.00068*cos(phase)*(sin(pi*u)**.40)
            x+=detail*sin(tt)*cos(phi)
            y-=detail*cos(tt)*cos(phi)
            z+=detail*sin(phi)
            vs.append((x,y,z))
    for j in range(rows-1):
        for k in range(n):
            a=j*n+k;b=j*n+(k+1)%n;fs.append((a,b,b+n,a+n))
    fs.append(tuple((rows-1)*n+k for k in range(n)))
    mesh_object('Hair | continuous slicked back sculpted cap',vs,fs,M['hair'],subsurf=1)
    # Fitted temple tapers complete the front hairline.
    for s in (-1,1):
        vv=[(s*.116,-.027,1.743),(s*.124,-.017,1.725),(s*.124,-.009,1.681),(s*.116,-.026,1.676),(s*.116,-.033,1.707)]
        mesh_object(('L' if s<0 else 'R')+' short sideburn',vv,[tuple(range(5))],M['hair'],subsurf=1)

def build_head():
    _head_surface()
    _ears()
    _eye(-1);_eye(1)
    _lips()
    _hair()
