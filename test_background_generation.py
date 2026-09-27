"""A pending background cannot replace newer scene/project work."""
import copy
import threading
import unittest
from object_directing import ObjectDirectorSession
from scene_composition import make_preset


class BackgroundGenerationIsolationTests(unittest.TestCase):
    def setUp(self):
        self.session = ObjectDirectorSession.__new__(ObjectDirectorSession)
        self.session.lock = threading.RLock()
        self.session.scene = {'gate': {'enabled': True}, **make_preset('Cozy living room')}
        self.session.project_revision = 0
        self.generated = make_preset('Winter plaza')

    def run_with_change(self, change):
        session, generated = self.session, self.generated
        class Generator:
            def generate(self, prompt):
                change()
                return generated
        return session.generate_scene('winter square', Generator())

    def test_motion_edit_does_not_discard_background(self):
        def motion_edit():
            self.session.project_revision += 1
            self.session.last_prompt = 'Jump and turn'
        self.assertEqual(self.run_with_change(motion_edit), self.generated)
        self.assertEqual(self.session.last_prompt, 'Jump and turn')

    def test_switch_to_identical_project_still_discards_background(self):
        original = self.session.scene_document()
        def project_switch():
            self.session.scene = copy.deepcopy(self.session.scene)
        with self.assertRaisesRegex(ValueError, 'Project changed'):
            self.run_with_change(project_switch)
        self.assertEqual(self.session.scene_document(), original)

    def test_scene_change_then_undo_still_discards_background(self):
        original = self.session.scene_document()
        def scene_changes():
            self.session.set_scene(make_preset('Enchanted grove'))
            self.session.set_scene(original)
        with self.assertRaisesRegex(ValueError, 'Project changed'):
            self.run_with_change(scene_changes)
        self.assertEqual(self.session.scene_document(), original)

    def test_direct_scene_change_is_also_protected(self):
        def scene_change():
            self.session.scene['lighting'] = 'sunset'
        with self.assertRaisesRegex(ValueError, 'Project changed'):
            self.run_with_change(scene_change)
        self.assertEqual(self.session.scene_document()['lighting'], 'sunset')


if __name__ == '__main__':
    unittest.main()
