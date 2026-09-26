"""Shared geometry helpers for the hand-built Patron character."""
import math
import bpy

M = {}
CHARACTER = None

def color(hexvalue):
    values = [int(hexvalue[i:i+2],16)/255 for i in (0,2,4)]
    return tuple(v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in values) + (1,)

def material(name, hexvalue, roughness=.5, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color(hexvalue)
    node = mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = color(hexvalue)
    node.inputs['Roughness'].default_value = roughness
    node.inputs['Metallic'].default_value = metallic
    return mat

def initialize():
    global CHARACTER
    CHARACTER = bpy.data.collections.new('PATRON | Character geometry')
    bpy.context.scene.collection.children.link(CHARACTER)
    palette = {
        'suit': ('4b3546',.78,0), 'shirt':('1c2026',.76,0),
        'lining':('2d2431',.68,0), 'seam':('493d45',.58,0),
        'button':('302731',.29,.18), 'skin':('c8967e',.49,0),
        'skin_shadow':('a87562',.56,0), 'lips':('986458',.5,0),
        'hair':('211b1d',.58,0), 'eye_white':('d7cdc0',.3,0),
        'iris':('61523e',.3,0), 'pupil':('11151a',.23,0),
        'nails':('d2a590',.4,0), 'leather':('18191e',.38,0),
        'sole':('12141a',.52,0),
    }
    for key, (hexvalue, rough, metal) in palette.items():
        M[key] = material(key,hexvalue,rough,metal)
    M['skin'].node_tree.nodes.get('Principled BSDF').inputs['Subsurface Weight'].default_value=.055
    M['suit'].node_tree.nodes.get('Principled BSDF').inputs['Sheen Weight'].default_value=.08
    for key in ('suit','shirt','hair','leather','skin'):
        M[key].node_tree.nodes.get('Principled BSDF').inputs['Specular IOR Level'].default_value=.25

def mesh_object(name, vertices, faces, mat, subsurf=1):
    mesh = bpy.data.meshes.new(name + ' | topology')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    CHARACTER.objects.link(obj)
    mesh.materials.append(M[mat] if isinstance(mat,str) else mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if subsurf:
        modifier = obj.modifiers.new('Subdivision | smooth crafted surface', 'SUBSURF')
        modifier.levels = subsurf
        modifier.render_levels = subsurf
    return obj

def ellipsoid(name, location, scale, mat, segments=48, rings=32):
    vertices = [(location[0],location[1],location[2]+scale[2])]
    for ring in range(1,rings):
        phi = math.pi*ring/rings
        for i in range(segments):
            angle = 2*math.pi*i/segments
            vertices.append((location[0]+scale[0]*math.sin(phi)*math.cos(angle),
                             location[1]+scale[1]*math.sin(phi)*math.sin(angle),
                             location[2]+scale[2]*math.cos(phi)))
    bottom = len(vertices)
    vertices.append((location[0],location[1],location[2]-scale[2]))
    faces=[]
    for i in range(segments):
        faces.append((0,1+i,1+(i+1)%segments))
    for ring in range(rings-2):
        a=1+ring*segments
        b=a+segments
        for i in range(segments):
            j=(i+1)%segments
            faces.append((a+i,b+i,b+j,a+j))
    a=1+(rings-2)*segments
    for i in range(segments):
        faces.append((a+i,bottom,a+(i+1)%segments))
    return mesh_object(name,vertices,faces,mat,subsurf=0)

def curve_tube(name, points, radius, mat, cyclic=False):
    curve=bpy.data.curves.new(name+' | curve','CURVE')
    curve.dimensions='3D'
    curve.resolution_u=12
    curve.bevel_depth=radius
    curve.bevel_resolution=3
    spline=curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points)-1)
    for point, coordinate in zip(spline.bezier_points,points):
        point.co=coordinate
        point.handle_left_type='AUTO'
        point.handle_right_type='AUTO'
    spline.use_cyclic_u=cyclic
    obj=bpy.data.objects.new(name,curve)
    CHARACTER.objects.link(obj)
    curve.materials.append(M[mat] if isinstance(mat,str) else mat)
    return obj
