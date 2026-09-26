import copy
import tempfile
import unittest
from unittest.mock import Mock
from adaptive_scene_generation import AdaptiveSceneGenerator, asset_system_prompt, fit_generated_assets
from scene_asset_library import AssetLibrary
from scene_environments import make_room
from scene_composition import validate_scene

class AdaptiveTests(unittest.TestCase):
    @staticmethod
    def overlapping_facade():
        return {'id':'facade','name':'Windowed facade','parts':[
            {'shape':'box','position':[0,0,0],'size':[1,1,.2],'color':[90,90,90]},
            {'shape':'box','position':[-.2,0,.08],'size':[.3,.15,.02],'color':[160,190,220],
             'repeat':{'count':[3,3,1],'step':[.2,.2,0]}}]}

    def test_generated_and_reused_assets_receive_geometry_review(self):
        from asset_quality import assess_assets
        flawed = self.overlapping_facade()
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            gateway.request_json.return_value = {'assets':[flawed]}
            generated = AdaptiveSceneGenerator(gateway, AssetLibrary(directory)).prepare_assets('a city')
            self.assertEqual(gateway.request_json.call_count, 1)
            self.assertFalse([i for i in assess_assets(generated) if i['severity']=='error'])
            reused = AdaptiveSceneGenerator(gateway, AssetLibrary(directory), [flawed]).prepared_assets
            self.assertFalse([i for i in assess_assets(reused) if i['severity']=='error'])
            self.assertEqual(flawed['parts'][1]['size'][0], .3)

    def test_custom_gateway_keeps_original_request_signature(self):
        class StrictGateway:
            def request_json(self, system, prompt, max_tokens):
                self.last_limit = max_tokens
                return {'assets':[AdaptiveTests.overlapping_facade()]}
        with tempfile.TemporaryDirectory() as directory:
            gateway = StrictGateway()
            AdaptiveSceneGenerator(gateway, AssetLibrary(directory)).prepare_assets('city')
            self.assertEqual(gateway.last_limit, 16000)

    def test_large_pack_prompt_and_eight_asset_persistence(self):
        """A full scene set survives validation, review, and library storage."""
        self.assertIn('6 to 8 reusable original assets', asset_system_prompt())
        self.assertIn('fewer than 180 total expanded shapes', asset_system_prompt())
        assets = []
        for i in range(8):
            asset = copy.deepcopy(self.overlapping_facade())
            asset['id'] = f'house-{i}'
            asset['name'] = f'House {i}'
            asset['parts'][0]['color'] = [90 + i, 90, 90]
            assets.append(asset)
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            gateway.request_json.return_value = {'assets': assets}
            library = AssetLibrary(directory)
            result = AdaptiveSceneGenerator(gateway, library).prepare_assets('large neighborhood')
            self.assertEqual(len(result), 8)
            self.assertEqual(len(library.load_all()), 8)
            self.assertEqual(gateway.request_json.call_count, 1)

    def test_city_layout_uses_original_buildings_without_second_model_call(self):
        from scene_environments import make_city
        source = make_city()
        assets = [a for a in source['assets'] if a['id'] in ('masonry','office')]
        assets = [dict(a,id='generated-'+a['id']) for a in assets]
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            result = AdaptiveSceneGenerator(gateway, AssetLibrary(directory), assets).generate('city street')
            gateway.request_json.assert_not_called()
            self.assertTrue(any(o['asset'].startswith('generated-') for o in result['objects']))
            self.assertEqual(validate_scene(result),result)
            self.assertIn('camera',result)

    def test_generated_overhang_is_fitted_without_losing_small_details(self):
        asset = {'id':'cabinet','name':'Cabinet','parts':[
            {'shape':'box','position':[0,0,0],'size':[1,1,1],'color':[100,80,60]},
            {'shape':'box','position':[0,0,.51],'size':[.1,.005,.03],'color':[240,220,180]}]}
        from asset_geometry import validate_assets
        with self.assertRaises(ValueError): validate_assets([asset])
        fitted = fit_generated_assets([asset])
        self.assertEqual(validate_assets(fitted), fitted)
        self.assertEqual(len(fitted[0]['parts']),2)
        bad = copy.deepcopy(asset)
        bad['parts'][1]['shape']='script'
        with self.assertRaises(ValueError): fit_generated_assets([bad])

    def test_preparation_persists_geometry_and_layout_reuses_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            library = AssetLibrary(directory)
            gateway = Mock()
            room = make_room()
            gateway.request_json.return_value = {'assets': room['assets']}
            generator = AdaptiveSceneGenerator(gateway, library)
            assets = generator.prepare_assets('a designed room')
            self.assertTrue(all(a['id'].startswith('a-') for a in assets))
            self.assertEqual(len(library.load_all()),len(assets))
            mapping = dict(zip((a['id'] for a in room['assets']), (a['id'] for a in assets)))
            layout = copy.deepcopy(room)
            layout['assets'] = []
            for obj in layout['objects']: obj['asset'] = mapping[obj['asset']]
            gateway.request_json.return_value = layout
            result = generator.generate('an abstract theatrical set')
            self.assertEqual(gateway.request_json.call_count, 2)
            self.assertEqual(result['assets'], assets)
            self.assertEqual(validate_scene(result), result)

    def test_bad_geometry_retries_once_and_never_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            gateway.request_json.return_value = {'assets':[{'id':'bad','name':'bad','parts':[]}]}
            library = AssetLibrary(directory)
            with self.assertRaises(ValueError):
                AdaptiveSceneGenerator(gateway, library).prepare_assets('a room')
            self.assertEqual(gateway.request_json.call_count,2)
            self.assertEqual(library.load_all(),[])

    def test_invalid_json_response_gets_one_compact_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            gateway.request_json.side_effect = [
                ValueError('Object gateway returned an invalid JSON scene'),
                {'assets': [self.overlapping_facade()]},
            ]
            assets = AdaptiveSceneGenerator(gateway, AssetLibrary(directory)).prepare_assets('neighborhood')
            self.assertEqual(len(assets), 1)
            self.assertEqual(gateway.request_json.call_count, 2)
            repair_system = gateway.request_json.call_args_list[1].args[0]
            self.assertIn('Keep the geometry concise', repair_system)

    def test_transport_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            gateway.request_json.side_effect = ValueError('Object gateway connection failed; check configuration and retry')
            with self.assertRaisesRegex(ValueError, 'connection failed'):
                AdaptiveSceneGenerator(gateway, AssetLibrary(directory)).prepare_assets('neighborhood')
            self.assertEqual(gateway.request_json.call_count, 1)

    def test_unknown_asset_reference_retries_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = Mock()
            room = make_room()
            room['objects'][0]['asset']='missing'
            gateway.request_json.return_value=room
            with self.assertRaises(ValueError):
                AdaptiveSceneGenerator(gateway, AssetLibrary(directory), room['assets']).generate('abstract set')
            self.assertEqual(gateway.request_json.call_count,2)


    def test_facade_at_unit_boundary_gets_room_for_repair(self):
        from asset_quality import assess_assets
        asset={'id':'edge','name':'Edge facade','parts':[
            {'shape':'box','position':[0,0,0],'size':[.8,.8,.8],'color':[120,120,120]},
            {'shape':'box','position':[-.2,0,.48],'size':[.12,.15,.01],'color':[100,160,200],
             'repeat':{'count':[3,1,1],'step':[.2,0,0]}},
            {'shape':'box','position':[0,0,.495],'size':[.8,.8,.01],'color':[10,10,10]}]}
        self.assertTrue(any(i['code']=='facade_occlusion' for i in assess_assets([asset])))
        reviewed=AdaptiveSceneGenerator._review_assets([asset])
        self.assertFalse(any(i['severity']=='error' for i in assess_assets(reviewed)))
        self.assertEqual(reviewed,AdaptiveSceneGenerator._review_assets(reviewed))

if __name__=='__main__': unittest.main()
