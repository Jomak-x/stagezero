"""Tests for reusable procedural scene assets."""

import json
import tempfile
import unittest
from pathlib import Path

from scene_asset_library import AssetLibrary


def asset(identifier="draft", name="Wooden crate", color=None):
    return {
        "id": identifier,
        "name": name,
        "parts": [{
            "shape": "box",
            "position": [0, 0, 0],
            "size": [0.8, 0.8, 0.8],
            "color": color or [145, 95, 55],
        }],
    }


class AssetLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "scene-assets"
        self.library = AssetLibrary(self.folder)

    def test_save_and_load_reuses_geometry_and_keeps_assets_detached(self):
        stored = self.library.save([asset()])[0]
        self.assertRegex(stored["id"], r"^a-[0-9a-f]{16}$")
        self.assertEqual(stored, self.library.get(stored["id"]))
        self.assertEqual([stored], AssetLibrary(self.folder).load_all())

        stored["name"] = "Changed in caller"
        stored["parts"][0]["color"][0] = 0
        reused = self.library.save([asset("different-input-id", "Other name")])[0]
        self.assertEqual(reused["name"], "Wooden crate")
        self.assertEqual(reused["parts"][0]["color"], [145, 95, 55])
        self.assertEqual(len(list(self.folder.glob("*.json"))), 1)

    def test_same_input_id_with_changed_geometry_creates_new_asset(self):
        first = self.library.save([asset("crate")])[0]
        second = self.library.save([asset("crate", color=[40, 80, 120])])[0]
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual({first["id"], second["id"]}, {item["id"] for item in self.library.list_assets()})

    def test_invalid_pack_does_not_change_existing_assets(self):
        first = self.library.save([asset()])[0]
        files_before = {path.name: path.read_bytes() for path in self.folder.glob("*.json")}
        invalid = asset("bad")
        invalid["parts"][0]["size"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            self.library.save([asset("new", color=[1, 2, 3]), invalid])
        self.assertEqual(files_before, {path.name: path.read_bytes() for path in self.folder.glob("*.json")})
        self.assertEqual(first, self.library.get(first["id"]))

    def test_corrupt_file_and_unsafe_id_are_not_used(self):
        stored = self.library.save([asset()])[0]
        path = self.folder / (stored["id"] + ".json")
        corrupt = json.dumps({"id": stored["id"], "name": "tampered", "parts": []})
        path.write_text(corrupt)
        self.assertIsNone(self.library.get(stored["id"]))
        self.assertEqual(self.library.load_all(), [])
        with self.assertRaises(ValueError):
            self.library.save([asset()])
        self.assertEqual(path.read_text(), corrupt)
        with self.assertRaises(ValueError):
            self.library.get("../other")


if __name__ == "__main__":
    unittest.main()
