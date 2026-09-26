"""Assemble a newly modeled Blender character and render review views."""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
parser=argparse.ArgumentParser()
parser.add_argument('--parts',default='body,head,extremities')
parser.add_argument('--view',choices=['hero','front','side','back','face'],default='hero')
parser.add_argument('--size',type=int,default=900)
parser.add_argument('--final',action='store_true')
parser.add_argument('--extra-views',default='')
parser.add_argument('--output-dir',type=Path)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])

bpy.ops.wm.read_factory_settings(use_empty=True)
import common
common.initialize()
for part in args.parts.split(','):
    module=importlib.import_module(part)
    getattr(module,'build_'+part)()
    print('BUILT',part,flush=True)

scene=bpy.context.scene
scene.unit_settings.system='METRIC'
scene.unit_settings.length_unit='METERS'
scene.render.engine='BLENDER_EEVEE'
scene.render.resolution_x=args.size
scene.render.resolution_y=int(args.size*1.25)
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.film_transparent=False
scene.world=bpy.data.worlds.new('Studio ambient')
scene.world.use_nodes=True
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.52,.56,.64,1)
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.35

studio=bpy.data.collections.new('STUDIO | cameras and lights')
scene.collection.children.link(studio)

def to_studio(obj):
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    studio.objects.link(obj)

bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.002))
floor=bpy.context.object
floor.name='Studio floor'
floor.data.materials.append(common.material('Studio warm grey','c4c2c1',.8))
to_studio(floor)

def aim(obj,target):
    quat=(Vector(target)-obj.location).to_track_quat('-Z','Y')
    obj.rotation_euler=quat.to_euler()
    return quat

for name,position,power,size,tint in [
    ('Key softbox',(-3,-4,4.5),550,3.0,(1,.9,.8)),
    ('Fill softbox',(3,-2,2.8),230,3.0,(.8,.88,1)),
    ('Rim softbox',(1,2,3.5),600,2.5,(1,.93,.84)),
]:
    bpy.ops.object.light_add(type='AREA',location=position)
    light=bpy.context.object
    light.name=name
    light.data.energy=power
    light.data.shape='DISK'
    light.data.size=size
    light.data.color=tint
    aim(light,(0,0,1.0))
    to_studio(light)

views={
    'hero':((-2.9,-6,2.8),(0,-.01,.96),2.30),
    'front':((0,-6,1.05),(0,0,1.0),2.20),
    'side':((6,0,1.05),(0,0,1.0),2.20),
    'back':((0,6,1.15),(0,0,1.0),2.20),
    'face':((-1.25,-4,2.02),(0,-.01,1.705),.52),
}
position,target,scale=views[args.view]
bpy.ops.object.camera_add(location=position)
camera=bpy.context.object
camera.name='Portrait camera'
camera.data.type='ORTHO'
camera.data.ortho_scale=scale
quat=aim(camera,target)
scene.camera=camera
to_studio(camera)

for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.shading.color_type='MATERIAL'
            area.spaces.active.region_3d.view_rotation=quat
            area.spaces.active.region_3d.view_location=(0,0,.95)
            area.spaces.active.region_3d.view_distance=3.2
            area.spaces.active.region_3d.view_perspective='ORTHO'

bpy.ops.object.select_all(action='DESELECT')
objects=list(common.CHARACTER.objects)
for obj in objects:
    if obj.type=='MESH':
        bpy.context.view_layer.objects.active=obj
        obj.select_set(True)
        break

out=ROOT/'assets/characters/patron-blender' if args.final else HERE.parent/'review'
if args.output_dir:
    out=args.output_dir.resolve()
out.mkdir(parents=True,exist_ok=True)
scene.render.filepath=str(out/(args.view+'.png'))
if args.final:
    blend=out/'patron.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend),compress=True)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active=objects[0]
    bpy.ops.object.convert(target='MESH')
    bpy.ops.export_scene.gltf(filepath=str(out/'patron.glb'),export_format='GLB',
                              use_selection=True,export_apply=True,export_yup=True,
                              export_animations=False)
    metadata={'objects':len(objects),'parts':args.parts,'method':'Blender procedural surface modeling',
              'reference':'Gemini_Generated_Image_6vmhe66vmhe66vmh.jpg',
              'blender_version':bpy.app.version_string,'rigged':False,
              'blender_coordinates':'Z-up, front -Y, meters','glb_coordinates':'Y-up, front +Z, meters'}
    (out/'model-info.json').write_text(json.dumps(metadata,indent=2))
else:
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'review.blend'),compress=True)
scene.render.filepath=str(out/(args.view+'.png'))
print('RENDER',scene.render.filepath,flush=True)
bpy.ops.render.render(write_still=True)
for view in filter(None,args.extra_views.split(',')):
    position,target,scale=views[view]
    camera.location=position
    camera.data.ortho_scale=scale
    aim(camera,target)
    scene.render.filepath=str(out/(view+'.png'))
    print('RENDER',scene.render.filepath,flush=True)
    bpy.ops.render.render(write_still=True)
