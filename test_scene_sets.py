import unittest
from scene_sets import make_market, make_workshop
from scene_asset_composition import environment_kind, compose_architectural_scene
from scene_composition import generate_recipe, validate_scene

class DenseSetTests(unittest.TestCase):
    def test_dense_staging_valid_and_actor_aisle_clear(self):
        for factory in (make_market,make_workshop):
            doc=factory()
            self.assertEqual(validate_scene(doc),doc)
            self.assertGreaterEqual(len(doc['objects']),30)
            self.assertEqual(factory(),doc)
            for obj in doc['objects']:
                if obj['position'][1]<0 or obj['position'][1]>4:continue
                x,_,z=obj['position'];sx,_,sz=obj['size']
                if abs(obj.get('yaw',0))==90:sx,sz=sz,sx
                self.assertTrue(abs(x)-sx/2>1.2 or abs(z)-sz/2>1.2, obj['name'])

    def test_specific_scene_intent_wins_over_city_or_room_words(self):
        for prompt,kind in [('residential city street','residential'),('city market square','market'),('warehouse workshop interior','workshop')]:
            self.assertEqual(environment_kind(prompt),kind)
            self.assertEqual(generate_recipe(prompt)['version'],3)

    def test_stall_variants_are_placed_and_export_embeds_them(self):
        base=next(a for a in make_market()['assets'] if a['id']=='set-stall')
        import copy
        assets=[]
        for i in range(3):
            a=copy.deepcopy(base);a.update(id=f'custom-{i}',name=f'Vendor stall {i}');assets.append(a)
        doc=compose_architectural_scene('city market square',assets)
        refs={o['asset'] for o in doc['objects']}
        self.assertTrue({a['id'] for a in assets}<=refs)
        self.assertTrue({a['id'] for a in assets}<={a['id'] for a in doc['assets']})

    def test_street_lamp_is_not_matched_as_tree(self):
        import copy
        lamp=copy.deepcopy(next(a for a in make_market()['assets'] if a['id']=='street-lamp'))
        lamp.update(id='generated-lamp',name='Ornate street lamp')
        doc=make_market(generated=[lamp])
        refs=[o['asset'] for o in doc['objects']]
        self.assertIn('street-tree',refs)
        self.assertEqual(refs.count('generated-lamp'),4)

    def test_workshop_storage_does_not_replace_shipping_crates(self):
        import copy
        base=next(a for a in make_workshop()['assets'] if a['id']=='set-crate')
        pack=[]
        for identifier,name in [('rack','Tall shelving with crates'),('crates','Stacked shipping crates on pallet'),('control','Industrial control cabinet')]:
            a=copy.deepcopy(base);a.update(id=identifier,name=name);pack.append(a)
        doc=make_workshop(generated=pack)
        crates=[o for o in doc['objects'] if o['asset']=='crates']
        self.assertEqual(len(crates),4)
        control=next(o for o in doc['objects'] if o['asset']=='control')
        racks=[o for o in doc['objects'] if o['asset']=='rack']
        for rack in racks:
            self.assertGreater(abs(rack['position'][0]-control['position'][0]),(rack['size'][2]+control['size'][0])/2)

if __name__=='__main__':unittest.main()
