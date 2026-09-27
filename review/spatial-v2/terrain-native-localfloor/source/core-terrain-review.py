import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scene_interaction_geometry import SceneInteractionGeometry
base=Path('/private/tmp/core-terrain-native-v2-results')
g=SceneInteractionGeometry.from_scene(json.load(open('/private/tmp/terrain-route-corrected-temple.json')))
im=Image.new('RGB',(1500,1040),'white');draw=ImageDraw.Draw(im)
colors=['#3355aa','#317baa','#279791','#33a655','#759e25','#ad8325']
for row,directory in enumerate(['terrain-native-v2','terrain-native-v2-localfloor']):
 p=base/directory;r=json.load(open(p/'report.json'));names=r['joint_names'];parents=r['parents']
 for col,variant in enumerate(['height_sparse','height_dense','height_foot']):
  d=np.load(p/(variant+'__seed33.npz'));pos=d['positions'];c=next(c for c in r['cases'] if c['name']==variant+'__seed33');m=c['metrics']
  ox=col*500;oy=row*500
  draw.text((ox+15,oy+10),('Absolute Y' if row==0 else 'Rigid local floor')+' / '+variant+' / seed33',fill='black')
  draw.text((ox+15,oy+28),f"Worst toe {m['min_toe_surface_clearance_m']:.3f}m; below -5cm: {m['toe_penetration_fraction_below_minus_5cm']:.0%}",fill='black')
  def xy(z,y):return (ox+20+(-z-5.0)*120,oy+475-(y-1.5)*120)
  support=[]
  for z in np.linspace(-5.1,-8.55,350):
   expected=np.interp(-z,-np.array(r['route'])[:,2],np.array(r['route'])[:,1]);y=g.support_height(0,z,expected,max_step_up=.4,max_drop=.4)
   if y is not None:support.append(xy(z,y))
  draw.line(support,fill='#101010',width=4)
  for fi,f in enumerate([0,31,63,95,127,159]):
   for j,parent in enumerate(parents):
    if parent is None:continue
    a,b=pos[f,[j,names.index(parent)]]
    draw.line([xy(a[2],a[1]),xy(b[2],b[1])],fill=colors[fi],width=2)
   for j in [names.index('LeftToeBase'),names.index('RightToeBase')]:
    x,y=xy(pos[f,j,2],pos[f,j,1]);draw.ellipse([x-3,y-3,x+3,y+3],fill='red')
  draw.text((ox+15,oy+482),'Unmodified native FK; black = exact centerline support; red = toes',fill='black')
draw.text((20,1020),'Geometry diagnostic only: frames 0,31,63,95,127,159. No root or joint corrections; coordinate frame restoration is rigid.',fill='black')
im.save(base/'fk-contact-sheet.png')
