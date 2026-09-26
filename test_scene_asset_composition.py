import unittest
from scene_asset_composition import compose_architectural_scene
from scene_environments import make_city, make_room
from scene_composition import validate_scene

class CompositionTests(unittest.TestCase):
    def test_city_never_stacks_complete_buildings(self):
        assets=[a for a in make_city()['assets'] if a['id'] in ('masonry','office','storefront')]
        for a in assets:a['id']='new-'+a['id']
        doc=compose_architectural_scene('an urban city street',assets)
        self.assertEqual(doc,validate_scene(doc))
        towers=[o for o in doc['objects'] if o['asset'].startswith('new-')]
        self.assertGreater(len(towers),4)
        for obj in towers:self.assertAlmostEqual(obj['position'][1],obj['size'][1]/2)
        self.assertTrue(all(o['name']!='Right upper stories' for o in doc['objects']))

    def test_unrelated_asset_is_not_scaled_into_a_building(self):
        plant=next(a for a in make_room()['assets'] if a['id']=='indoor-plant')
        self.assertIsNone(compose_architectural_scene('city street',[plant]))

    def test_desk_gets_working_height_not_coffee_table_dimensions(self):
        desk=next(a for a in make_room()['assets'] if a['id']=='coffee-table')
        desk.update(id='new-desk',name='Walnut writing desk')
        doc=compose_architectural_scene('a quiet studio room',[desk])
        obj=next(o for o in doc['objects'] if o['asset']=='new-desk')
        self.assertEqual(obj['size'],[1.5,.78,.75])
        self.assertAlmostEqual(obj['position'][1],.39)

    def test_surface_decoration_is_above_floor_base(self):
        for factory,key in ((make_city,'asphalt'),(make_city,'sidewalk'),(make_room,'floorboards')):
            asset=next(a for a in factory()['assets'] if a['id']==key)
            base=asset['parts'][0]
            top=base['position'][1]+base['size'][1]/2
            for detail in asset['parts'][1:]:
                self.assertGreater(detail['position'][1]-detail['size'][1]/2,top)


class PlacementRegressionTests(unittest.TestCase):
    def test_drafting_table_and_ceiling_pendant_get_distinct_roles(self):
        import copy
        original=next(a for a in make_room()['assets'] if a['id']=='coffee-table')
        desk=copy.deepcopy(original);desk.update(id='draft',name='Oak drafting table')
        lamp=copy.deepcopy(original);lamp.update(id='pendant',name='Brass ceiling pendant')
        doc=compose_architectural_scene('a design studio room',[desk,lamp])
        placed={o['asset']:o for o in doc['objects']}
        self.assertEqual(placed['draft']['size'],[1.5,.78,.75])
        self.assertGreater(placed['pendant']['position'][1],2.5)

    def test_skyline_keeps_central_street_open(self):
        doc=make_city()
        towers=[o for o in doc['objects'] if o['asset'] in ('masonry','office','brownstone')]
        self.assertGreater(len(towers),4)
        for obj in towers:
            self.assertGreater(abs(obj['position'][0])-obj['size'][0]/2,1.8)

if __name__=='__main__':unittest.main()
