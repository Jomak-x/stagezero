"""Regression coverage for curved-mesh budgets and large scene round trips."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from asset_geometry import _primitive, triangle_count
from scene_composition import validate_scene, encode_scene, MAX_SCENE_FILE_BYTES
from test_scene_performance import make_session as _make_session

def make_session(doc):
    session=_make_session(copy.deepcopy(doc))
    session.scene['gate']={'enabled':False}
    session.project_revision=0
    return session



def document():
    asset={'id':'balls','name':'Repeated spheres','parts':[
        {'shape':'sphere','position':[-.4,0,-.4],'size':[.01,.01,.01],'color':[100,150,180],
         'repeat':{'count':[16,1,31],'step':[.05,0,.026]}}]}
    objects=[{'id':f'obj-{i}','name':'Test prop','kind':'custom','asset':'balls','position':[i*2,1,0],
              'size':[1,2,1],'color':[255,255,255],
              'interaction':{'action':'none','trigger':'none','radius':0}} for i in range(3)]
    return {'version':3,'name':'Triangle boundary','assets':[asset],'objects':objects,'effects':[],'lighting':'neutral'}


class ScaleLimitTests(unittest.TestCase):
    def test_cost_matches_actual_primitive_tessellation(self):
        for shape in ('box','sphere','cylinder','cone'):
            recipe={'parts':[{'shape':shape,'repeat':{'count':[2,3,4]}}]}
            self.assertEqual(triangle_count(recipe),len(_primitive(shape)[1])*24)

    def test_curved_budget_boundary_rejects_without_replacing_scene(self):
        doc=document()
        self.assertEqual(triangle_count(doc['assets'][0])*3,249984)
        session=make_session(doc)
        previous=session.scene_document();revision=session.project_revision
        bad=copy.deepcopy(doc)
        bad['assets'][0]['parts'].append({'shape':'sphere','position':[0,.2,0],'size':[.01,.01,.01],'color':[100,150,180]})
        with self.assertRaisesRegex(ValueError,'250000 generated triangles'):session.set_scene(bad)
        self.assertEqual(session.scene_document(),previous)
        self.assertEqual(session.project_revision,revision)

    def test_large_pretty_and_compact_exports_round_trip(self):
        doc=document();base=doc['assets'][0]
        part={'shape':'box','position':[0,0,0],'size':[.3,.4,.5],'color':[100,150,180],
              'rotation':[0,0,0],'repeat':{'count':[1,1,1],'step':[0,0,0]}}
        doc['assets']=[];doc['objects']=[]
        for i in range(16):
            doc['assets'].append({'id':f'asset-{i}','name':'Detailed storage assembly','parts':[copy.deepcopy(part) for _ in range(64)]})
        template=document()['objects'][0]
        for i in range(64):
            obj=copy.deepcopy(template);obj.update(id=f'obj-{i}',asset=f'asset-{i%16}');doc['objects'].append(obj)
        clean=validate_scene(doc)
        pretty=json.dumps(clean,indent=2).encode()
        self.assertGreater(len(pretty),500000)
        self.assertLess(len(pretty),MAX_SCENE_FILE_BYTES)
        for payload in (pretty,encode_scene(clean)):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'scene.json';path.write_bytes(payload)
                session=make_session(document());session.load_objects(path)
                self.assertEqual(session.scene_document(),clean)

if __name__=='__main__':unittest.main()
