"""Offline checks for conservative reference-to-UV detail transfer."""

import unittest

import numpy as np
from PIL import Image
import trimesh

from character_texture_detail import (
    _bilinear, _pad_projected_atlas, _reference_pixels, enhance_front_texture,
)


def layered_mesh():
    """Three UV-separated quads: visible front, rear, and hidden front."""
    vertices, faces, uv = [], [], []
    for index, (z, reverse) in enumerate(((0.2, False), (-0.2, True), (0.0, False))):
        start = len(vertices)
        vertices.extend([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]])
        tris = [[0, 2, 1], [0, 3, 2]] if reverse else [[0, 1, 2], [0, 2, 3]]
        faces.extend([[start + value for value in tri] for tri in tris])
        left, right = index / 3, (index + 1) / 3
        uv.extend([[left, 0], [right, 0], [right, 1], [left, 1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGBA", (96, 64), (40, 40, 40, 255)),
            metallicFactor=0.0,
        )
    )
    return mesh


def cutout(color=(230, 70, 20, 255)):
    image = Image.new("RGBA", (64, 64))
    image.paste(color, (8, 8, 56, 56))
    return image


class ReferenceDetailTests(unittest.TestCase):
    def test_narrow_foreground_survives_without_borrowing_background_rgb(self):
        reference = Image.new("RGBA", (64, 64), (0, 255, 255, 0))
        # A one-pixel opaque feature would disappear under a 3x3 erosion.
        reference.paste((190, 120, 80, 255), (31, 10, 32, 54))
        original = np.asarray(reference).copy()
        pixels = _reference_pixels(reference)
        np.testing.assert_array_equal(pixels[32, 31], [190, 120, 80, 255])
        np.testing.assert_array_equal(pixels[32, 30], [190, 120, 80, 255])
        self.assertEqual(pixels[32, 29, 3], 224)  # Zero transfer beyond its bound.
        np.testing.assert_array_equal(np.asarray(reference), original)

    def test_edge_extension_has_no_source_when_alpha_is_uncertain(self):
        pixels = _reference_pixels(Image.new("RGBA", (64, 64), (255, 0, 255, 240)))
        self.assertFalse(pixels[:, :, 3].any())

    def test_padding_removes_filtered_uv_edge_without_touching_other_island(self):
        atlas = np.full((16, 24, 4), (240, 240, 240, 255), dtype=np.uint8)
        atlas[4:12, 4:10, :3] = (120, 50, 20)
        atlas[4:12, 14:20, :3] = (20, 50, 120)
        occupied = np.zeros((16, 24), dtype=bool)
        occupied[4:12, 4:10] = True
        occupied[4:12, 14:20] = True
        projected = np.zeros_like(occupied)
        projected[4:12, 4:10] = True
        original = atlas.copy()
        # Filtering at u=edge otherwise mixes the old pale bake into the face.
        before = _bilinear(atlas.astype(float), np.array([[4.0, 7.5]]))[0, :3]
        self.assertGreater(before[0], 120)
        count = _pad_projected_atlas(atlas, occupied, projected)
        after = _bilinear(atlas.astype(float), np.array([[4.0, 7.5]]))[0, :3]
        np.testing.assert_array_equal(after, [120, 50, 20])
        np.testing.assert_array_equal(atlas[occupied], original[occupied])
        np.testing.assert_array_equal(atlas[:, :, 3], original[:, :, 3])
        # The adjacent rear island owns its own nearest gutter.
        np.testing.assert_array_equal(atlas[7, 13], original[7, 13])
        self.assertGreater(count, 0)

    def test_only_visible_front_changes_and_input_is_unchanged(self):
        mesh = layered_mesh()
        original = np.asarray(mesh.visual.material.baseColorTexture).copy()
        result, report = enhance_front_texture(mesh, cutout(), visibility_size=64)
        pixels = np.asarray(result.visual.material.baseColorTexture)
        self.assertTrue(report["applied"], report)
        self.assertGreater(report["silhouette_iou"], 0.99)
        self.assertGreater(pixels[32, 16, 0], 190)
        np.testing.assert_array_equal(pixels[:, 32:], original[:, 32:])
        np.testing.assert_array_equal(np.asarray(mesh.visual.material.baseColorTexture), original)
        np.testing.assert_array_equal(result.vertices, mesh.vertices)
        np.testing.assert_array_equal(result.faces, mesh.faces)

    def test_neutral_reference_does_not_change_texture(self):
        mesh = layered_mesh()
        result, report = enhance_front_texture(mesh, cutout((40, 40, 40, 255)), visibility_size=64)
        self.assertFalse(report["applied"])
        self.assertEqual(report["changed_texels"], 0)
        np.testing.assert_array_equal(np.asarray(result.visual.material.baseColorTexture),
                                      np.asarray(mesh.visual.material.baseColorTexture))

    def test_image_x_and_uv_y_orientation(self):
        reference = cutout()
        reference.paste((10, 220, 30, 255), (8, 8, 32, 32))
        result, report = enhance_front_texture(layered_mesh(), reference, visibility_size=64)
        pixels = np.asarray(result.visual.material.baseColorTexture)
        self.assertTrue(report["applied"])
        self.assertGreater(pixels[16, 8, 1], pixels[16, 8, 0])
        self.assertGreater(pixels[48, 8, 0], pixels[48, 8, 1])
        self.assertGreater(pixels[16, 24, 0], pixels[16, 24, 1])

    def test_missing_real_alpha_is_rejected(self):
        for reference in (Image.new("RGB", (64, 64)), Image.new("RGBA", (64, 64)),
                          Image.new("RGBA", (64, 64), (255, 0, 0, 255))):
            _, report = enhance_front_texture(layered_mesh(), reference, visibility_size=64)
            self.assertFalse(report["applied"])
            self.assertIn("cutout", report["reason"])

    def test_silhouette_mismatch_is_rejected(self):
        reference = Image.new("RGBA", (64, 64))
        reference.paste((255, 0, 0, 255), (8, 8, 56, 16))
        reference.paste((255, 0, 0, 255), (8, 48, 56, 56))
        _, report = enhance_front_texture(layered_mesh(), reference, visibility_size=64)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "reference_pose_or_silhouette_mismatch")

    def test_zero_strength_is_noop(self):
        _, report = enhance_front_texture(layered_mesh(), cutout(), strength=0, visibility_size=64)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "zero_strength")


if __name__ == "__main__":
    unittest.main()
