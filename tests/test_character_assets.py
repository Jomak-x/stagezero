"""Behavioral tests for bounded, self-contained GLB character imports."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from character_assets import AssetLimits, AssetValidationError, DEFAULT_LIMITS, import_glb, inspect_glb, load_character_asset
from tests.glb_fixtures import base_document_and_binary, make_glb, make_humanoid_glb, make_static_glb


def unpack_glb(data: bytes) -> tuple[dict, bytes]:
    json_size = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + json_size])
    bin_start = 20 + json_size + 8
    return document, data[bin_start:]


class CharacterAssetTests(unittest.TestCase):
    def test_default_file_cap_is_500_decimal_megabytes_without_expanding_other_caps(self) -> None:
        self.assertEqual(DEFAULT_LIMITS.max_file_bytes, 500_000_000)
        self.assertEqual(DEFAULT_LIMITS.max_json_bytes, 4 * 1024 * 1024)
        self.assertEqual(DEFAULT_LIMITS.max_expanded_bytes, 256 * 1024 * 1024)
        self.assertEqual(DEFAULT_LIMITS.max_image_pixels, 32_000_000)

    def test_static_preview_preserves_original_material_bytes(self) -> None:
        data = make_static_glb()
        asset = inspect_glb(data, display_name='<script>alert(1)</script>')
        self.assertEqual(asset.kind, "static")
        self.assertEqual(asset.status, "static_preview")
        self.assertEqual((asset.vertex_count, asset.triangle_count), (3, 1))
        self.assertEqual(asset.glb_bytes, data)
        self.assertIn("pbrMetallicRoughness", asset.document["materials"][0])
        self.assertNotIn("<", asset.display_name)
        np.testing.assert_allclose(asset.read_accessor(0)[2], [0, 0.4, 0])
        np.testing.assert_allclose(asset.bounds, [[-0.2, 0, 0], [0.2, 0.4, 0]])

    def test_two_distinct_humanoid_rigs_are_mapping_required(self) -> None:
        for scheme in ("g1", "mixamo"):
            with self.subTest(scheme=scheme):
                asset = inspect_glb(make_humanoid_glb(scheme))
                self.assertEqual((asset.kind, asset.status), ("skinned", "mapping_required"))
                self.assertEqual(len(asset.skins[0]["joints"]), 12)
                self.assertEqual(asset.nodes[1].parent, None)
                self.assertAlmostEqual(asset.nodes[2].world_matrix[1, 3],
                                       1.24 if scheme == "g1" else 1.488)
                self.assertEqual(asset.read_accessor(3).shape, (3, 4))
                self.assertEqual(asset.image_pixels, 4 if scheme == "mixamo" else 0)
                if scheme == "mixamo":
                    self.assertEqual(asset.document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"], {"index": 0})
                np.testing.assert_allclose(asset.read_accessor(4).sum(axis=1), [1, 1, 1])
                np.testing.assert_allclose(asset.read_accessor(5)[0],
                                           [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1 * (1 if scheme == "g1" else 1.2), 0, 1])

    def test_import_uses_generated_id_and_revalidates_on_load(self) -> None:
        data = make_humanoid_glb()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = import_glb(data, root, display_name="My hero")
            self.assertEqual((root / asset.asset_id / "asset.glb").read_bytes(), data)
            self.assertEqual(load_character_asset(root, asset.asset_id).sha256, asset.sha256)
            self.assertEqual(import_glb(data, root).asset_id, asset.asset_id)
            with self.assertRaisesRegex(AssetValidationError, "invalid asset ID"):
                load_character_asset(root, "../../etc/passwd")
            (root / asset.asset_id / "asset.glb").write_bytes(b"corrupt")
            with self.assertRaises(AssetValidationError):
                load_character_asset(root, asset.asset_id)

    def test_rejects_header_chunk_and_resource_attacks(self) -> None:
        valid = make_static_glb()
        doc, binary = unpack_glb(valid)
        altered = dict(doc)
        altered["extensionsRequired"] = ["KHR_draco_mesh_compression"]
        attacks = [
            (valid[:8] + struct.pack("<I", len(valid) + 4) + valid[12:], "header"),
            (valid[:12] + struct.pack("<I", 0xFFFFFFFF) + valid[16:], "chunk"),
            (valid[:16] + b"JUNK" + valid[20:], "JSON"),
            (make_glb(altered, binary), "extensions"),
        ]
        for data, expected in attacks:
            with self.subTest(expected=expected), self.assertRaisesRegex(AssetValidationError, expected):
                inspect_glb(data)
        doc["buffers"][0]["uri"] = "https://example.test/evil.bin"
        with self.assertRaisesRegex(AssetValidationError, "external"):
            inspect_glb(make_glb(doc, binary))
        del doc["buffers"][0]["uri"]
        doc["images"] = [{"uri": "../../secret.png"}]
        with self.assertRaisesRegex(AssetValidationError, "external"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(valid)
        del doc["accessors"]
        with self.assertRaisesRegex(AssetValidationError, "missing accessors"):
            inspect_glb(make_glb(doc, binary))
        with self.assertRaisesRegex(AssetValidationError, "size limit"):
            inspect_glb(valid, limits=AssetLimits(max_file_bytes=len(valid) - 1))

    def test_rejects_optional_extension_payloads_and_morph_targets(self) -> None:
        doc, binary = base_document_and_binary()
        # Three.js can honor this payload even when extensionsUsed omits it.
        doc["nodes"][0]["extensions"] = {"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 0}}}
        with self.assertRaisesRegex(AssetValidationError, "extension payloads"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = base_document_and_binary()
        doc["extensionsUsed"] = ["EXT_mesh_gpu_instancing"]
        with self.assertRaisesRegex(AssetValidationError, "extensionsUsed"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = base_document_and_binary()
        doc["meshes"][0]["primitives"][0]["targets"] = [{"POSITION": 999999}]
        with self.assertRaisesRegex(AssetValidationError, "morph target"):
            inspect_glb(make_glb(doc, binary))

    def test_selected_scene_must_display_mesh_and_only_it_sets_bounds(self) -> None:
        doc, binary = base_document_and_binary()
        doc["nodes"].append({"mesh": 0, "translation": [100, 0, 0]})
        doc["scenes"].append({"nodes": [1]})
        asset = inspect_glb(make_glb(doc, binary))
        np.testing.assert_allclose(asset.bounds, [[-0.2, 0, 0], [0.2, 0.4, 0]])
        doc["scenes"][0]["nodes"] = []
        with self.assertRaisesRegex(AssetValidationError, "selected GLB scene has no mesh"):
            inspect_glb(make_glb(doc, binary))
        doc["scene"] = 1
        asset = inspect_glb(make_glb(doc, binary))
        np.testing.assert_allclose(asset.bounds, [[99.8, 0, 0], [100.2, 0.4, 0]])

    def test_skinned_bounds_use_joint_world_not_mesh_node_transform(self) -> None:
        doc, binary = unpack_glb(make_humanoid_glb())
        doc["nodes"][0]["translation"] = [100, 0, 0]
        asset = inspect_glb(make_glb(doc, binary))
        np.testing.assert_allclose(asset.bounds, [[-0.2, 0, 0], [0.2, 0.4, 0]], atol=1e-5)
        doc["scenes"][0]["nodes"] = [0]
        with self.assertRaisesRegex(AssetValidationError, "omits a skinned mesh joint"):
            inspect_glb(make_glb(doc, binary))

    def test_rejects_accessor_bounds_stride_and_non_finite_data(self) -> None:
        for key, malformed in (("componentType", []), ("type", {})):
            doc, binary = base_document_and_binary()
            doc["accessors"][0][key] = malformed
            with self.subTest(key=key), self.assertRaisesRegex(AssetValidationError, "unsupported component type or shape"):
                inspect_glb(make_glb(doc, binary))
        doc, binary = base_document_and_binary()
        doc["accessors"][0]["count"] = 100
        with self.assertRaisesRegex(AssetValidationError, "exceeds bufferView"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = base_document_and_binary()
        doc["bufferViews"][0]["byteStride"] = 4
        with self.assertRaisesRegex(AssetValidationError, "stride"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = base_document_and_binary()
        binary = struct.pack("<f", float("nan")) + binary[4:]
        with self.assertRaisesRegex(AssetValidationError, "non-finite"):
            inspect_glb(make_glb(doc, binary))

    def test_interleaved_accessor_stride_reads_correctly(self) -> None:
        doc, _ = base_document_and_binary()
        vertices = [(-0.2, 0, 0), (0.2, 0, 0), (0, 0.4, 0)]
        interleaved = b"".join(struct.pack("<6f", *point, 0, 0, 1) for point in vertices)
        binary = interleaved + struct.pack("<3H", 0, 1, 2)
        doc["bufferViews"] = [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(interleaved), "byteStride": 24},
            {"buffer": 0, "byteOffset": len(interleaved), "byteLength": 6},
        ]
        doc["accessors"][0]["bufferView"] = 0
        doc["accessors"][1]["bufferView"] = 0
        doc["accessors"][1]["byteOffset"] = 12
        doc["accessors"][2]["bufferView"] = 1
        asset = inspect_glb(make_glb(doc, binary))
        np.testing.assert_allclose(asset.read_accessor(0), vertices)
        np.testing.assert_allclose(asset.read_accessor(1), [(0, 0, 1)] * 3)

    def test_rejects_cyclic_hierarchy_and_bad_transform(self) -> None:
        doc, binary = base_document_and_binary()
        doc["nodes"][0]["children"] = [0]
        with self.assertRaises(AssetValidationError):
            inspect_glb(make_glb(doc, binary))
        doc, binary = base_document_and_binary()
        doc["nodes"][0]["translation"] = [0, 0, float("inf")]
        with self.assertRaises(ValueError):
            make_glb(doc, binary)
        doc["nodes"][0]["translation"] = [0, 0, 0]
        doc["nodes"][0]["scale"] = [1, 0, 1]
        with self.assertRaisesRegex(AssetValidationError, "singular"):
            inspect_glb(make_glb(doc, binary))
        doc["nodes"][0]["scale"] = [10 ** 500, 1, 1]
        with self.assertRaisesRegex(AssetValidationError, "non-finite"):
            inspect_glb(make_glb(doc, binary))

    def test_rejects_bad_skin_joints_weights_and_inverse_bind(self) -> None:
        doc, binary = unpack_glb(make_humanoid_glb())
        doc["skins"][0]["joints"] = []
        with self.assertRaisesRegex(AssetValidationError, "no joints"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(make_humanoid_glb())
        doc["meshes"][0]["primitives"][0]["attributes"].pop("WEIGHTS_0")
        with self.assertRaisesRegex(AssetValidationError, "WEIGHTS_0"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(make_humanoid_glb())
        view = doc["bufferViews"][4]
        offset = view["byteOffset"]
        binary = binary[:offset] + struct.pack("<4f", 0.2, 0, 0, 0) + binary[offset + 16:]
        with self.assertRaisesRegex(AssetValidationError, "sum to one"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(make_humanoid_glb())
        doc["accessors"][5]["count"] = 11
        with self.assertRaisesRegex(AssetValidationError, "inverse bind"):
            inspect_glb(make_glb(doc, binary))

    def test_rejects_unbounded_expanded_geometry(self) -> None:
        data = make_humanoid_glb()
        with self.assertRaisesRegex(AssetValidationError, "expanded"):
            inspect_glb(data, limits=AssetLimits(max_expanded_bytes=200))
        doc, binary = base_document_and_binary()
        doc["nodes"].append({"mesh": 0, "translation": [1, 0, 0]})
        doc["scenes"][0]["nodes"].append(1)
        with self.assertRaisesRegex(AssetValidationError, "mesh instances"):
            inspect_glb(make_glb(doc, binary), limits=AssetLimits(max_vertices=5))

    def test_rejects_bad_embedded_image_and_material_reference(self) -> None:
        doc, binary = unpack_glb(make_humanoid_glb("mixamo"))
        doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] = 3
        with self.assertRaisesRegex(AssetValidationError, "baseColorTexture.index"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(make_humanoid_glb("mixamo"))
        doc["meshes"][0]["primitives"][0]["attributes"].pop("TEXCOORD_0")
        with self.assertRaisesRegex(AssetValidationError, "lacks TEXCOORD_0"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(make_humanoid_glb("mixamo"))
        view = doc["bufferViews"][6]
        offset = view["byteOffset"]
        binary = binary[:offset] + b"bad image" + binary[offset + 9:]
        with self.assertRaisesRegex(AssetValidationError, "invalid embedded image"):
            inspect_glb(make_glb(doc, binary))
        doc, binary = unpack_glb(make_humanoid_glb("mixamo"))
        view = doc["bufferViews"][6]
        offset = view["byteOffset"] + 45  # A PNG chunk payload/CRC byte, not its signature.
        binary = binary[:offset] + bytes([binary[offset] ^ 0x01]) + binary[offset + 1:]
        with self.assertRaisesRegex(AssetValidationError, "invalid embedded image"):
            inspect_glb(make_glb(doc, binary))
        with self.assertRaisesRegex(AssetValidationError, "image dimensions exceed limit"):
            inspect_glb(make_humanoid_glb("mixamo"), limits=AssetLimits(max_image_pixels=3))


if __name__ == "__main__":
    unittest.main()
