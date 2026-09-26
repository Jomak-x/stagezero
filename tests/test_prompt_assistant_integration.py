"""Motion submission must clarify ambiguous direction without starting inference."""
import unittest

import test_studio_ui as fixtures


class PromptAssistantIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.StudioUITests.setUp(self)
        # The shared lightweight fake predates the public folder property.
        self.ui.prompt_assistant.folder.expand_by_default = False
        self.ui.action_assistant.folder.expand_by_default = False
    tearDown = fixtures.StudioUITests.tearDown
    _seed_take = fixtures.StudioUITests._seed_take
    _seed_segmented_take = fixtures.StudioUITests._seed_segmented_take

    def test_ambiguous_new_motion_opens_assistant_without_backend_request(self):
        self.ui.prompt.edit('turn back')
        self.ui.generate.click()
        self.assertFalse(self.session.busy)
        self.assertIn('Clarify', self.session.status)
        self.assertEqual(self.ui.prompt.value, 'turn back')
        self.assertTrue(self.ui.prompt_assistant.folder.visible)

    def test_ambiguous_action_edit_preserves_existing_take(self):
        take = self._seed_segmented_take()
        original = take.positions.copy()
        self.ui._begin_action_edit(take, 0, 'replace')
        self.ui.update()
        self.ui.action_prompt.edit('turn back')
        self.ui.save_action.click()
        self.assertFalse(self.session.busy)
        self.assertIs(self.session.takes[take.id], take)
        self.assertTrue((take.positions == original).all())
        self.assertIn('Clarify', self.session.status)

    def test_explicit_turnaround_can_generate(self):
        self.ui.prompt.edit('Turn the whole body 180 degrees in place to face the opposite direction.')
        self.ui.generate.click()
        self.assertTrue(self.session.busy)
        self.assertNotIn('Clarify', self.session.status)

    def test_assistant_hidden_for_static_character(self):
        self.session.set_character_motion_enabled(False)
        self.ui.update()
        self.assertFalse(self.ui.prompt_assistant.folder.visible)
        self.assertFalse(self.ui.action_assistant.folder.visible)

    def test_apply_rechecks_source_and_does_not_generate(self):
        self.ui.prompt.edit('turn back')
        context = self.ui._assistant_context(False)
        self.ui.prompt.edit('Wave with your left hand')
        accepted = self.ui.prompt_assistant.apply('Turn 180 degrees in place.', context, 'turn back')
        self.assertFalse(accepted)
        self.assertEqual(self.ui.prompt.value, 'Wave with your left hand')
        context = self.ui._assistant_context(False)
        accepted = self.ui.prompt_assistant.apply('Turn 180 degrees in place.', context,
                                                 'Wave with your left hand')
        self.assertTrue(accepted)
        self.assertEqual(self.ui.prompt.value, 'Turn 180 degrees in place.')
        self.assertFalse(self.session.busy)


if __name__ == '__main__':
    unittest.main()
