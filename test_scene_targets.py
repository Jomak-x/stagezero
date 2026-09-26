import copy
import json
import unittest
from scene_targets import validate_targets,resolve_targets,TargetLayer
from scene_composition import make_preset,validate_scene,encode_scene
from test_scene_scale_limits import make_session
from test_scene_performance import make_layer

class TargetTests(unittest.TestCase):
    def fixture(self):
        d=make_preset('City boulevard');o=d['objects'][4]
        d['targets']=[{'id':'roof','name':'Roof landing','object_id':o['id'],'kind':'landing','local_position':[0,.5,0]}]
        return d,o
    def test_roundtrip_and_rotated_attachment(self):
        d,o=self.fixture();o.update(position=[10,4,20],size=[4,6,8],yaw=90)
        d['targets'][0]['local_position']=[.5,.5,0]
        clean=validate_scene(d)
        self.assertEqual(json.loads(encode_scene(clean)),clean)
        self.assertEqual(resolve_targets(clean['targets'],clean['objects'])[0]['position'],[10,7,18])
    def test_deleted_prop_prunes_target_and_edit_keeps_attachment(self):
        d,o=self.fixture();session=make_session(d)
        session.edit_object(o['id'],position=[5,10,3])
        self.assertEqual(len(session.scene_document()['targets']),1)
        self.assertEqual(resolve_targets(session.scene_document()['targets'],session.scene_document()['objects'])[0]['position'][0],5)
        session.remove_object(o['id'])
        self.assertEqual(session.scene_document()['targets'],[])
    def test_replacement_clears_old_targets(self):
        d,_=self.fixture();session=make_session(d);session.set_scene(make_preset('Designed apartment'))
        self.assertNotIn('targets',session.scene_document())
    def test_bad_targets_rejected(self):
        d,_=self.fixture()
        for key,value in [('object_id','missing'),('kind','execute'),('local_position',[0,float('nan'),0]),('local_position',[0,1,0])]:
            bad=copy.deepcopy(d);bad['targets'][0][key]=value
            with self.assertRaises(ValueError):validate_scene(bad)
        bad=copy.deepcopy(d);bad['targets']*=2
        with self.assertRaises(ValueError):validate_scene(bad)
    def test_markers_removed_and_hidden_bundle_is_empty(self):
        d,_=self.fixture();session=make_session(d)
        self.assertEqual(session.object_states()['targets'],[])
        revision=session.project_revision
        session.show_scene_targets=True
        self.assertGreater(session.project_revision,revision)
        self.assertEqual(len(session.object_states()['targets']),1)
        layer,scene=make_layer()
        scene.add_icosphere=lambda name,**kwargs:scene.add_glb(name,b'test-marker',wxyz=(1,0,0,0))
        markers=TargetLayer(layer.server)
        markers.update(d['targets'],d['objects']);self.assertEqual(len(markers.handles),1)
        markers.update([],d['objects']);self.assertEqual(markers.handles,{})

if __name__=='__main__':unittest.main()
