"""Validation, pure playback, and scene-node lifecycle checks for effects."""

import copy
import math
import unittest

import numpy as np

from scene_effects import EFFECT_KINDS, MAX_EFFECTS, EffectSceneLayer, evaluate_effects, make_effect, validate_effects


class EffectDataTests(unittest.TestCase):
    def test_all_defaults_are_canonical_and_distinct(self):
        effects = [make_effect(kind, i) for i, kind in enumerate(EFFECT_KINDS)]
        self.assertEqual(effects, validate_effects(effects))
        self.assertEqual(len({e["id"] for e in effects}), len(effects))
        self.assertEqual(len({tuple(e["color"]) for e in effects}), len(effects))

    def test_rejects_unknown_fields_duplicate_ids_and_invalid_units(self):
        good = make_effect("smoke", 0)
        mutations = (
            lambda x: x.update(script="exec()"),
            lambda x: x.update(kind="fire"),
            lambda x: x.update(id="bad id"),
            lambda x: x.update(position=[0, math.nan, 0]),
            lambda x: x.update(position=[21, 0, 0]),
            lambda x: x.update(size=[0, 1, 1]),
            lambda x: x.update(size=[9, 1, 1]),
            lambda x: x.update(color=[256, 0, 0]),
            lambda x: x.update(intensity=1.01),
            lambda x: x.update(seed=-1),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                bad = copy.deepcopy(good)
                mutate(bad)
                validate_effects([bad])
        with self.assertRaises(ValueError):
            validate_effects([good, copy.deepcopy(good)])
        with self.assertRaises(ValueError):
            validate_effects([make_effect("snow", i) for i in range(MAX_EFFECTS + 1)])
        clean = validate_effects([good])[0]
        good["position"][0] = 20
        self.assertNotEqual(clean["position"], good["position"])

    def test_rejects_invalid_playhead(self):
        for value in (-1, math.inf, math.nan, "2", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_effects([make_effect("rain", 0)], value)


class EffectPlaybackTests(unittest.TestCase):
    def test_scrub_is_exactly_reproducible_and_particle_counts_stay_bounded(self):
        effects = [make_effect(kind, i) for i, kind in enumerate(EFFECT_KINDS)]
        baseline = evaluate_effects(effects, 2.25)
        evaluate_effects(effects, 8.0)
        replay = evaluate_effects(effects, 2.25)
        self.assertEqual(len(baseline), len(effects))
        for first, second in zip(baseline, replay):
            self.assertEqual(first["id"], second["id"])
            self.assertTrue(128 <= len(first["points"]) <= 256)
            self.assertEqual(first["points"].shape, first["colors"].shape)
            self.assertEqual(first["points"].dtype, np.float32)
            self.assertEqual(first["colors"].dtype, np.uint8)
            self.assertTrue(np.isfinite(first["points"]).all())
            self.assertLess(np.abs(first["points"]).max(), 30)
            np.testing.assert_array_equal(first["points"], second["points"])
            np.testing.assert_array_equal(first["colors"], second["colors"])
            if "segments" in first:
                self.assertEqual(first["segments"].shape[1:], (2, 3))
                self.assertTrue(np.isfinite(first["segments"]).all())
                np.testing.assert_array_equal(first["segments"], second["segments"])

    def test_time_animates_every_effect_and_seed_changes_pattern(self):
        for kind in EFFECT_KINDS:
            with self.subTest(kind=kind):
                effect = make_effect(kind, 0)
                a = evaluate_effects([effect], 0.0)[0]["points"]
                b = evaluate_effects([effect], 0.5)[0]["points"]
                self.assertFalse(np.array_equal(a, b))
                effect["seed"] = 88
                c = evaluate_effects([effect], 0.0)[0]["points"]
                self.assertFalse(np.array_equal(a, c))

    def test_zero_intensity_has_no_visible_particles_or_ring(self):
        effect = make_effect("portal", 0)
        effect["intensity"] = 0
        state = evaluate_effects([effect], 3.0)[0]
        self.assertEqual(state["points"].shape, (0, 3))
        self.assertNotIn("segments", state)


class _Handle:
    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.removed = False

    def remove(self):
        self.removed = True


class _Scene:
    def __init__(self):
        self.created = []

    def add_point_cloud(self, name, points, colors, **kwargs):
        handle = _Handle(name=name, points=points, colors=colors, **kwargs)
        self.created.append(handle)
        return handle

    def add_line_segments(self, name, points, colors, **kwargs):
        handle = _Handle(name=name, points=points, colors=colors, **kwargs)
        self.created.append(handle)
        return handle


class _Server:
    def __init__(self):
        self.scene = _Scene()


class EffectSceneLayerTests(unittest.TestCase):
    def test_cinematic_descriptor_updates_in_place_and_switches_back_to_particles(self):
        server = _Server()
        layer = EffectSceneLayer(server)
        blast = make_effect('explosion', 0, position=(1, 2, 3))
        layer.update([blast], 1.25)
        handle = layer.handles[blast['id']]['cinematic_explosion']
        self.assertEqual(handle.name, '/effects/explosion-0/cinematic_explosion')
        self.assertEqual(handle.points.shape, (4, 3))
        self.assertEqual(handle.points.dtype, np.float32)
        np.testing.assert_allclose(handle.points[0], (1.25, .75, 0))
        np.testing.assert_allclose(handle.points[1], blast['size'])
        np.testing.assert_allclose(handle.points[2], np.asarray(blast['color']) / 255)
        self.assertEqual(handle.position, (1., 2., 3.))
        layer.update([blast], 2.5)
        self.assertIs(layer.handles[blast['id']]['cinematic_explosion'], handle)
        self.assertEqual(handle.points[0, 0], 2.5)
        rain = make_effect('rain', 0)
        rain['id'] = blast['id']
        layer.update([rain], 3.0)
        self.assertTrue(handle.removed)
        self.assertIn('points', layer.handles[blast['id']])

    def test_reuses_handles_when_seeking_and_cleans_removed_effects(self):
        server = _Server()
        layer = EffectSceneLayer(server)
        portal = make_effect("portal", 0)
        snow = make_effect("snow", 1)
        layer.update([portal, snow], 0.0)
        self.assertEqual(len(server.scene.created), 3)
        prior = {key: value.copy() for key, value in layer.handles.items()}
        layer.update([portal, snow], 2.0)
        self.assertEqual(len(server.scene.created), 3)
        self.assertIs(layer.handles[portal["id"]]["points"], prior[portal["id"]]["points"])
        layer.update([snow], 1.0)
        self.assertEqual(set(layer.handles), {snow["id"]})
        self.assertTrue(all(h.removed for h in prior[portal["id"]].values()))
        self.assertFalse(prior[snow["id"]]["points"].removed)
        layer.update([], 0.0)
        self.assertEqual(layer.handles, {})
        self.assertTrue(all(h.removed for h in server.scene.created))


if __name__ == "__main__":
    unittest.main()
