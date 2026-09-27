"""Actor count must survive both entry points and asynchronous cancellation."""
from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from scene_actor_intent import ActorIntent, ActorPreflight, classify_actor_intent
from studio_ui import CREATE, REPLACE, SET_DURATION
from tests import test_story_controls as story
import test_studio_ui as studio
from test_live_motion import wait_until


class ClassifierTests(TestCase):
    def test_preserves_raw_prompt_and_counts_including_implicit_partners(self):
        for prompt, count in [('  Two people shake hands.\n', 2),
                              ('A couple hug.', 2), ('Three friends dance.', 3),
                              ('Walk for 120 seconds and wave twice.', 1)]:
            with self.subTest(prompt=prompt):
                provider = Mock(return_value={'actor_count': count, 'reason': 'performers'})
                self.assertEqual(classify_actor_intent(prompt, provider=provider).count, count)
                self.assertEqual(provider.call_args.args[1], prompt)

    def test_invalid_count_is_not_single(self):
        for value in ('2', True, 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                classify_actor_intent('direction', provider=lambda *_: {'actor_count': value, 'reason': ''})
        self.assertTrue(ActorIntent(None).error)
        self.assertTrue(ActorIntent(4).error)

    def test_gate_double_submit_cancellation_and_late_results(self):
        started, release, finished = Event(), Event(), Event()
        def classify(_):
            started.set()
            release.wait(2)
            finished.set()
            return ActorIntent(2)
        gate = ActorPreflight(classify)
        self.assertTrue(gate.start('raw', ('scene', 1)))
        self.assertTrue(started.wait(1))
        self.assertFalse(gate.start('again', ('scene', 1)))
        gate.cancel()
        release.set()
        self.assertTrue(finished.wait(1))
        self.assertIsNone(gate.poll(('scene', 1)))
        self.assertFalse(gate.pending)

    def test_gate_failure_and_stale_context_never_default_single(self):
        for classifier, context in [(Mock(side_effect=RuntimeError('offline')), ('scene', 1)),
                                    (lambda _: ActorIntent(2), ('scene', 2))]:
            gate = ActorPreflight(classifier)
            gate.start('raw', ('scene', 1))
            result = []
            def finish():
                value = gate.poll(context)
                if value is not None:
                    result.append(value)
                    return True
                return False
            wait_until(finish)
            self.assertIsNone(result[0][0])
            self.assertTrue(result[0][1])


class FullSceneRoutingTests(TestCase):
    setUp = story.StoryControlsTests.setUp

    def submit(self, count, prompt='  Two friends greet each other.\n', length='Auto'):
        self.controls.on_generate_cast = Mock(return_value=True)
        view = self.controls.open(self.client)
        view.prompt.value, view.length.value = prompt, length
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(count)):
            view.generate.click(self.client)
        return view

    def test_two_and_three_preserve_original_and_never_activate_g1(self):
        for count in (2, 3):
            prompt = f'  {count} friends greet each other.\n'
            view = self.submit(count, prompt)
            self.controls.on_generate_cast.assert_called_once_with(prompt, None, self.client, actor_count=count)
            self.assertTrue(view.modal.closed)
            self.assertFalse(self.controls.workflow.jobs)
            self.assertEqual(self.handoffs, [])
            self.assertEqual(self.core.deactivations, 0)
            self.assertFalse(self.controls.sidebar_status.visible)
            self.assertEqual(self.controls._route_status, '')

    def test_single_long_story_keeps_original_duration_and_workflow(self):
        prompt = '  One person walks, waves, and rests.\n'
        view = self.submit(1, prompt, '120 seconds')
        job = next(iter(self.controls.workflow.jobs.values()))
        self.assertEqual((job['prompt'], job['seconds']), (prompt, 120))
        self.controls.on_generate_cast.assert_not_called()
        self.assertFalse(view.modal.closed)

    def test_explicit_multiactor_length_rejected_without_ignoring_it(self):
        view = self.submit(2, length='60 seconds')
        self.assertIn('Length: Auto', view.error)
        self.assertEqual(view.length.value, '60 seconds')
        self.controls.on_generate_cast.assert_not_called()
        self.assertFalse(self.controls.workflow.jobs)

    def test_uncertain_unsupported_and_failure_do_not_generate(self):
        for count in (None, 4):
            view = self.submit(count)
            self.assertTrue(view.error)
            self.assertFalse(self.controls.workflow.jobs)
            self.controls.on_generate_cast.assert_not_called()
        with patch('scene_actor_intent.classify_actor_intent', side_effect=RuntimeError('offline')):
            view.generate.click(self.client)
        self.assertIn('Could not check', view.error)
        self.assertFalse(self.controls.workflow.jobs)

    def test_closing_rejected_scene_clears_global_error_for_other_work(self):
        view = self.submit(10)
        self.assertIn('Scenes support', self.controls.sidebar_status.content)
        view.close.click(self.client)
        self.controls.update()
        self.assertNotIn('Scenes support', self.controls.sidebar_status.content)
        self.assertFalse(self.controls.sidebar_status.visible)

    def test_missing_callback_fails_closed(self):
        view = self.controls.open(self.client)
        view.prompt.value = 'Two people greet'
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)):
            view.generate.click(self.client)
        self.assertIn('unavailable', view.error)
        self.assertFalse(self.controls.workflow.jobs)

    def test_cancel_close_text_length_and_scene_changes_discard_late_completion(self):
        for change in ('cancel', 'close', 'text', 'length', 'scene'):
            with self.subTest(change=change):
                release, started, finished = Event(), Event(), Event()
                view = self.controls.open(self.client)
                view.prompt.value = 'Two people greet'
                callback = self.controls.on_generate_cast = Mock(return_value=True)
                def classify(_):
                    started.set()
                    release.wait(2)
                    finished.set()
                    return ActorIntent(2)
                view.actor_check.classifier = classify
                with patch('scene_actor_intent.Thread', Thread):
                    view.generate.click(self.client)
                    self.assertTrue(started.wait(1))
                    self.assertTrue(view.generate.disabled)
                    view.generate.click(self.client)
                    if change == 'cancel': view.cancel.click(self.client)
                    elif change == 'close': view.close.click(self.client)
                    elif change == 'text': view.prompt.value += ' and wave'
                    elif change == 'length': view.length.value = '30 seconds'
                    else: self.session.project_revision += 1
                    self.controls.update()
                    release.set()
                    self.assertTrue(finished.wait(1))
                    self.controls.update()
                callback.assert_not_called()
                self.assertFalse(self.controls.workflow.jobs)

    def test_cancelling_new_preflight_preserves_previous_selected_running_job(self):
        for status in ('planning', 'running', 'speech_failed'):
            with self.subTest(status=status):
                self.controls.ids.clear()
                self.controls.workflow.jobs.clear()
                old = self.controls.workflow.submit('One person waves', seconds=15)
                self.controls.workflow.jobs[old]['status'] = status
                self.controls.ids['Previous scene'] = old
                view = self.controls.open(self.client)
                view.jobs.value = 'Previous scene'
                view.prompt.value = 'Two people greet'
                release, started, finished = Event(), Event(), Event()
                def classify(_):
                    started.set()
                    release.wait(2)
                    finished.set()
                    return ActorIntent(2)
                view.actor_check.classifier = classify
                with patch('scene_actor_intent.Thread', Thread):
                    view.generate.click(self.client)
                    self.assertTrue(started.wait(1))
                    view.cancel.click(self.client)
                    release.set()
                    self.assertTrue(finished.wait(1))
                    self.controls.update()
                self.assertEqual(self.controls.workflow.jobs[old]['status'], status)
                self.assertEqual(len(self.controls.workflow.jobs), 1)

    def test_multi_person_action_edit_is_rejected_without_g1_activation(self):
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        view.action_prompt.value = 'Two people shake hands'
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)):
            view.edit.click(self.client)
        self.assertIn('cannot update', view.error)
        self.assertFalse(self.session.calls)
        self.assertFalse(self.handoffs)


class MotionRoutingTests(TestCase):
    setUp = studio.StudioUITests.setUp
    tearDown = studio.StudioUITests.tearDown
    _seed_segmented_take = studio.StudioUITests._seed_segmented_take
    _seed_take = studio.StudioUITests._seed_take

    def test_create_routes_original_before_prompt_assistant_can_rewrite(self):
        self.ui.on_generate_cast = Mock(return_value=True)
        self.ui.prompt_assistant.refiner = Mock(side_effect=AssertionError('Must not rewrite cast'))
        prompt = '  Two friends walk toward each other and shake hands.\n'
        self.ui.prompt.edit(prompt)
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)):
            self.ui.generate.click()
        args = self.ui.on_generate_cast.call_args
        self.assertEqual(args.args[:2], (prompt, None))
        self.assertEqual(args.kwargs, {'actor_count': 2})
        self.ui.prompt_assistant.refiner.assert_not_called()
        self.assertFalse(self.session.busy)
        self.assertIn('2 performers', self.session.status)

    def test_previously_improved_direction_still_routes_original_cast(self):
        original = '  Two friends greet each other.\n'
        self.ui.prompt.edit('A person waves.')
        self.ui.prompt_assistant._applied = (self.ui._assistant_context(False), original,
                                            self.ui.prompt.value, self.ui._assistant_scene_context())
        self.ui.on_generate_cast = Mock(return_value=True)
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)) as classifier:
            self.ui.generate.click()
        classifier.assert_called_once_with(original)
        self.assertEqual(self.ui.on_generate_cast.call_args.args[0], original)
        self.assertFalse(self.backend.started.is_set())

    def test_timing_changes_do_not_discard_original_actor_provenance(self):
        original = 'Two friends greet each other.'
        self.ui.prompt.edit('A person waves.')
        self.ui.prompt_assistant._applied = (self.ui._assistant_context(False), original,
                                            self.ui.prompt.value, self.ui._assistant_scene_context())
        self.ui.on_generate_cast = Mock(return_value=True)
        self.ui.duration_mode.edit(SET_DURATION)
        self.ui.duration_seconds.edit('10')
        self.assertIsNone(self.ui.prompt_assistant._applied)
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)) as classifier:
            self.ui.generate.click()
        classifier.assert_called_once_with(original)
        self.assertIn('Length: Auto', self.session.status)
        self.assertFalse(self.backend.started.is_set())
        self.ui.on_generate_cast.assert_not_called()
        # A subsequent explicit text edit starts a new provenance chain.
        self.ui.prompt.edit('One person jumps.')
        self.assertEqual(self.ui._actor_original('motion'), 'One person jumps.')

    def test_action_timing_changes_preserve_original_actor_provenance(self):
        take = self._seed_segmented_take()
        self.ui._begin_action_edit(take, 0, 'replace')
        self.ui.update()
        original = 'Two people shake hands.'
        self.ui.action_prompt.edit('A person waves.')
        self.ui.action_assistant._applied = (self.ui._assistant_context(True), original,
                                            self.ui.action_prompt.value, self.ui._assistant_scene_context())
        self.ui.action_duration.edit('6')
        self.assertIsNone(self.ui.action_assistant._applied)
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)) as classifier:
            self.ui.save_action.click()
        classifier.assert_called_once_with(original)
        self.assertIn('cannot update', self.session.status)
        self.assertFalse(self.session.busy)
        self.assertIs(self.session.takes[take.id], take)

    def test_single_person_rewrite_still_generates_after_timing_change(self):
        original = 'One person walks forward.'
        self.ui.prompt.edit('Walk forward.')
        self.ui.prompt_assistant._applied = (self.ui._assistant_context(False), original,
                                            self.ui.prompt.value, self.ui._assistant_scene_context())
        self.ui.duration_mode.edit(SET_DURATION)
        self.ui.duration_seconds.edit('4')
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(1)) as classifier:
            self.ui.generate.click()
        classifier.assert_called_once_with(original)
        def started():
            self.ui.update()
            return self.backend.started.is_set()
        wait_until(started)

    def test_non_auto_multi_actor_request_is_not_shortened(self):
        self.ui.on_generate_cast = Mock(return_value=True)
        self.ui.prompt.edit('Three people dance together')
        self.ui.duration_mode.edit(SET_DURATION)
        self.ui.duration_seconds.edit('20')
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(3)):
            self.ui.generate.click()
        self.assertIn('Length: Auto', self.session.status)
        self.ui.on_generate_cast.assert_not_called()
        self.assertFalse(self.backend.started.is_set())

    def test_replace_and_action_edit_do_not_create_new_cast(self):
        take = self._seed_segmented_take()
        self.ui.on_generate_cast = Mock(return_value=True)
        self.ui.edit_action.value = REPLACE
        self.ui.prompt.value = 'Two people shake hands'
        self.ui.replace_time.value = '1'
        with patch('scene_actor_intent.classify_actor_intent', return_value=ActorIntent(2)):
            self.ui.generate.click()
            self.assertIn('cannot update', self.session.status)
            self.ui._begin_action_edit(take, 0, 'replace')
            self.ui.update()
            self.ui.action_prompt.edit('Two people shake hands')
            self.ui.save_action.click()
        self.assertIn('cannot update', self.session.status)
        self.ui.on_generate_cast.assert_not_called()
        self.assertFalse(self.session.busy)
        self.assertIs(self.session.takes[take.id], take)

    def test_cancel_and_changed_scene_prevent_late_motion_handoff(self):
        for cancel in (True, False):
            release, started, finished = Event(), Event(), Event()
            self.ui.on_generate_cast = Mock(return_value=True)
            self.ui.prompt.edit('Two people greet')
            def classify(_):
                started.set()
                release.wait(2)
                finished.set()
                return ActorIntent(2)
            self.ui._actor_check.classifier = classify
            with patch('scene_actor_intent.Thread', Thread):
                self.ui.generate.click()
                self.assertTrue(started.wait(1))
                if cancel: self.ui.cancel.click()
                else: self.session.scene.setdefault('objects', []).append({'id': 'new', 'name': 'box', 'position': [0, 0, 0]})
                self.ui.update()
                release.set()
                self.assertTrue(finished.wait(1))
                self.ui.update()
            self.ui.on_generate_cast.assert_not_called()
            self.assertFalse(self.backend.started.is_set())
