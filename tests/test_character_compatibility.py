"""Compatibility reports combine bounded GLB validation with rig usability."""
import copy
import unittest

from character_assets import inspect_glb
from character_compatibility import assess_character, inspect_character
from retargeting import detect_rig_profile
from tests.glb_fixtures import make_glb, make_humanoid_glb, make_static_glb


class CharacterCompatibilityTests(unittest.TestCase):
    def test_unskinned_mesh_has_preview_and_blender_guidance_without_mapping_action(self):
        asset, report = inspect_character(make_static_glb())
        self.assertIsNotNone(asset)
        self.assertEqual(report.status, 'static_preview')
        self.assertEqual(report.actions, ('Open static preview', 'Choose another file'))
        self.assertFalse(report.can_map)
        self.assertIn('cannot create bones or weights', ' '.join(report.warnings))

    def test_supported_rig_has_visual_quality_caveat(self):
        _, report = inspect_character(make_humanoid_glb('mixamo'))
        self.assertEqual(report.status, 'motion_ready')
        self.assertIsNotNone(report.retargeter)
        self.assertIn('does not guarantee', ' '.join(report.warnings))
        self.assertIn('Required humanoid bone mapping', report.checks)

    def test_hidden_scene_rig_cannot_make_the_selected_static_mesh_motion_ready(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        static_mesh = copy.deepcopy(document['meshes'][0])
        for primitive in static_mesh['primitives']:
            del primitive['attributes']['JOINTS_0']
            del primitive['attributes']['WEIGHTS_0']
        document['meshes'].append(static_mesh)
        static_node = len(document['nodes'])
        document['nodes'].append({'name': 'Selected static mesh', 'mesh': 1})
        document['scenes'].append({'nodes': [static_node]})
        document['scene'] = 1
        asset, report = inspect_character(make_glb(document, original.binary_chunk))
        self.assertEqual(asset.kind, 'skinned')  # File-wide metadata is insufficient.
        self.assertEqual(report.status, 'static_preview')
        self.assertFalse(report.can_map)
        self.assertIsNone(report.retargeter)
        self.assertIn('selected scene', ' '.join(report.reasons))
        document['scene'] = 0
        _, report = inspect_character(make_glb(document, original.binary_chunk))
        self.assertEqual(report.status, 'motion_ready')
        required = set(report.retargeter.profile.bones.values())
        self.assertTrue(required.issubset(set(document['skins'][0]['joints'])))

    def test_selected_skin_with_omitted_scene_joints_is_rejected_by_importer(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        document['scenes'][0]['nodes'] = [0]
        asset, report = inspect_character(make_glb(document, original.binary_chunk))
        self.assertIsNone(asset)
        self.assertEqual(report.status, 'unsupported')
        self.assertIn('omits a skinned mesh joint', ' '.join(report.reasons))

    def test_renamed_humanoid_can_request_mapping(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        for index, node in enumerate(document['nodes']):
            node['name'] = f'Joint{index}'
        _, report = inspect_character(make_glb(document, original.binary_chunk))
        self.assertEqual(report.status, 'mapping_required')
        self.assertTrue(report.can_preview)
        self.assertTrue(report.can_map)
        self.assertIn('Load rig mapping', report.actions)

    def test_incomplete_skin_cannot_be_repaired_by_a_mapping_file(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        document['skins'][0] = {'joints': [1], 'skeleton': 1}
        _, report = inspect_character(make_glb(document, original.binary_chunk))
        self.assertEqual(report.status, 'static_preview')
        self.assertFalse(report.can_map)
        self.assertIn('at least 12', ' '.join(report.reasons))

    def test_rig_with_nonuniform_transform_only_allows_static_preview(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        document['nodes'][1]['scale'] = [1, 2, 1]
        _, report = inspect_character(make_glb(document, original.binary_chunk))
        self.assertEqual(report.status, 'static_preview')
        self.assertIn('transforms', ' '.join(report.reasons))
        self.assertFalse(report.can_map)

    def test_minimal_recognized_rig_with_coincident_arm_joints_needs_preparation(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        profile = detect_rig_profile(original)
        document['nodes'][profile.bones['left_forearm']]['translation'] = [0, 0, 0]
        data = make_glb(document, original.binary_chunk)
        for mapping in (None, {'bones': dict(profile.bones)}):
            with self.subTest(mapping=mapping):
                _, report = inspect_character(data, mapping=mapping)
                self.assertEqual(report.status, 'static_preview')
                self.assertFalse(report.can_map)
                self.assertIn('arm rest length must be nonzero', ' '.join(report.reasons))
                self.assertIn('Blender', ' '.join(report.reasons))

    def test_alternative_skin_joint_can_repair_a_degenerate_detected_mapping(self):
        original = inspect_glb(make_humanoid_glb())
        document = copy.deepcopy(original.document)
        profile = detect_rig_profile(original)
        document['nodes'][profile.bones['left_forearm']]['translation'] = [0, 0, 0]
        alternate = len(document['nodes'])
        document['nodes'].append({'name': 'AlternateForearm', 'translation': [.3, 0, 0]})
        document['nodes'][profile.bones['left_upper_arm']]['children'].append(alternate)
        document['skins'][0]['joints'].append(alternate)
        del document['skins'][0]['inverseBindMatrices']
        data = make_glb(document, original.binary_chunk)
        _, report = inspect_character(data)
        self.assertEqual(report.status, 'mapping_required')
        self.assertTrue(report.can_map)
        bones = dict(profile.bones)
        bones['left_forearm'] = alternate
        _, report = inspect_character(data, mapping={'bones': bones})
        self.assertEqual(report.status, 'motion_ready')

    def test_corrupt_geometry_weights_and_hierarchy_are_unsupported(self):
        _, report = inspect_character(b'not a GLB')
        self.assertEqual(report.status, 'unsupported')
        self.assertEqual(report.actions, ('Choose another file',))
        self.assertFalse(report.can_preview)
        original = inspect_glb(make_humanoid_glb())
        for failure in ('weights', 'hierarchy', 'geometry'):
            with self.subTest(failure=failure):
                document = copy.deepcopy(original.document)
                if failure == 'weights':
                    del document['meshes'][0]['primitives'][0]['attributes']['WEIGHTS_0']
                elif failure == 'hierarchy':
                    document['nodes'][1]['children'].append(1)
                else:
                    document['meshes'][0]['primitives'][0]['mode'] = 1
                asset, report = inspect_character(make_glb(document, original.binary_chunk))
                self.assertIsNone(asset)
                self.assertEqual(report.status, 'unsupported')


if __name__ == '__main__':
    unittest.main()
