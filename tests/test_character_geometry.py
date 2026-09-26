"""Lab ground measurement follows rendered geometry rather than joint origins."""
import copy
import struct
import unittest

import numpy as np

from character_assets import inspect_glb
from character_geometry import ground_offset, posed_minimum_y
from tests.glb_fixtures import base_document_and_binary, make_glb, make_humanoid_glb


class CharacterGeometryTests(unittest.TestCase):
    def test_static_pose_uses_world_transform_and_selected_scene(self):
        doc, binary = base_document_and_binary()
        doc['nodes'].append({'mesh': 0, 'translation': [0, -100, 0]})
        doc['scenes'].append({'nodes': [1]})
        asset = inspect_glb(make_glb(doc, binary))
        matrices = np.stack([node.world_matrix for node in asset.nodes])
        matrices[0, 1, 3] = 2
        before = matrices.copy()
        self.assertAlmostEqual(posed_minimum_y(asset, matrices), 2)
        np.testing.assert_array_equal(matrices, before)

    def test_skin_uses_inverse_bind_and_ignores_mesh_node_transform(self):
        asset = inspect_glb(make_humanoid_glb())
        matrices = np.stack([node.world_matrix for node in asset.nodes])
        self.assertAlmostEqual(posed_minimum_y(asset, matrices), 0, places=6)
        matrices[0, 1, 3] = 100
        for joint in asset.skins[0]['joints']:
            matrices[joint, 1, 3] += 3
        self.assertAlmostEqual(posed_minimum_y(asset, matrices), 3, places=6)

    def test_unused_accessor_vertex_does_not_lower_ground(self):
        doc, binary = base_document_and_binary()
        # Draw only the upper vertex (a degenerate test triangle); lower buffer
        # vertices must not move the floor when they are not referenced.
        index_view = doc['bufferViews'][doc['accessors'][2]['bufferView']]
        start = index_view['byteOffset']
        binary = binary[:start] + struct.pack('<3H', 2, 2, 2)
        asset = inspect_glb(make_glb(copy.deepcopy(doc), binary))
        matrices = np.stack([node.world_matrix for node in asset.nodes])
        self.assertAlmostEqual(posed_minimum_y(asset, matrices), .4, places=6)

    def test_near_unit_weights_match_browser_normalization(self):
        base = inspect_glb(make_humanoid_glb())
        doc = copy.deepcopy(base.document)
        accessor = doc['accessors'][doc['meshes'][0]['primitives'][0]['attributes']['WEIGHTS_0']]
        view = doc['bufferViews'][accessor['bufferView']]
        start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        binary = bytearray(base.binary_chunk)
        for vertex in range(accessor['count']):
            struct.pack_into('<f', binary, start + vertex * 16, .99)
        asset = inspect_glb(make_glb(doc, bytes(binary)))
        matrices = np.stack([node.world_matrix for node in asset.nodes])
        for joint in asset.skins[0]['joints']:
            matrices[joint, 1, 3] += 3
        self.assertAlmostEqual(posed_minimum_y(asset, matrices), 3, places=6)

    def test_invalid_pose_is_rejected(self):
        doc, binary = base_document_and_binary()
        asset = inspect_glb(make_glb(doc, binary))
        with self.assertRaises(ValueError):
            posed_minimum_y(asset, np.eye(4))
        with self.assertRaises(ValueError):
            posed_minimum_y(asset, np.full((1, 4, 4), np.nan))

    def test_waist_origin_and_existing_ground_origin_at_different_scales(self):
        for bottom in (-.555, 0., .25):
            doc, binary = base_document_and_binary()
            doc['nodes'][0]['translation'] = [0, bottom, 0]
            asset = inspect_glb(make_glb(doc, binary))
            for scale in (.1, 1., 3.):
                with self.subTest(bottom=bottom, scale=scale):
                    offset = ground_offset(asset, scale=scale)
                    self.assertAlmostEqual(offset, -scale * bottom)
                    self.assertAlmostEqual(scale * bottom + offset, 0)
                    self.assertAlmostEqual(ground_offset(asset, scale=scale, floor_y=2.), 2. - scale * bottom)

    def test_ground_offset_uses_referenced_geometry_and_keeps_source(self):
        doc, binary = base_document_and_binary()
        start = doc['bufferViews'][doc['accessors'][2]['bufferView']]['byteOffset']
        binary = binary[:start] + struct.pack('<3H', 2, 2, 2)
        source = make_glb(doc, binary)
        asset = inspect_glb(source)
        self.assertAlmostEqual(ground_offset(asset), -.4, places=6)
        self.assertEqual(asset.glb_bytes, source)

    def test_ground_offset_rejects_invalid_placement(self):
        doc, binary = base_document_and_binary()
        asset = inspect_glb(make_glb(doc, binary))
        for scale in (0., -1., float('inf')):
            with self.assertRaises(ValueError):
                ground_offset(asset, scale=scale)
        with self.assertRaises(ValueError):
            ground_offset(asset, floor_y=float('nan'))


if __name__ == '__main__':
    unittest.main()
