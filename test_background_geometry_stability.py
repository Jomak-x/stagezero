"""Camera-independent checks for conflicting colored faces in background recipes."""

import itertools
import unittest

from scene_composition import PRESETS, make_preset


def _boxes(asset):
    for part_index, part in enumerate(asset['parts']):
        if part['shape'] != 'box' or any(part.get('rotation', ())):
            continue
        repeat = part.get('repeat', {'count': [1, 1, 1], 'step': [0, 0, 0]})
        for offset in itertools.product(*(range(n) for n in repeat['count'])):
            center = [part['position'][axis] + offset[axis] * repeat['step'][axis]
                      for axis in range(3)]
            bounds = [(center[axis] - part['size'][axis] / 2,
                       center[axis] + part['size'][axis] / 2) for axis in range(3)]
            yield part_index, part['color'], bounds


class BackgroundGeometryStabilityTests(unittest.TestCase):
    def test_colored_detail_faces_do_not_compete_for_the_same_depth(self):
        # Same-facing overlapping surfaces with different colors cause camera-
        # dependent z-fighting, even though every individual recipe is valid.
        # Opposite-facing seams and same-color joints are intentionally allowed.
        for preset in PRESETS:
            for asset in make_preset(preset, 0).get('assets', []):
                for first, second in itertools.combinations(_boxes(asset), 2):
                    i, color_a, a = first
                    j, color_b, b = second
                    if color_a == color_b:
                        continue
                    for axis in range(3):
                        if not all(min(a[k][1], b[k][1]) - max(a[k][0], b[k][0]) > 1e-8
                                   for k in range(3) if k != axis):
                            continue
                        for side in (0, 1):
                            self.assertGreater(abs(a[axis][side] - b[axis][side]), 1e-8,
                                               (preset, asset['id'], i, j, axis, side))


if __name__ == '__main__':
    unittest.main()
