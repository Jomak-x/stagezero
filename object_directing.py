"""UI-independent object editing integration for the directing controller."""
import copy
import json
from numbers import Integral
from directing import DirectorSession
from takes import decode_project
from scene_objects import validate_objects, evaluate_objects
from object_generation import generate_local, validate_prompt
from scene_composition import validate_scene, generate_recipe, MAX_SCENE_FILE_BYTES
from scene_effects import validate_effects


class ObjectDirectorSession(DirectorSession):
    def _prepare_generation(self, prompt, take, stop):
        from scene_motion import plan_scene_motion
        start = self.generation_start_positions(take, stop)
        scene = copy.deepcopy(self.scene)
        if take is not None and stop and scene.get('objects'):
            states = evaluate_objects(scene['objects'], take.positions, stop - 1, hand_indices=(25, 33))
            positions = {state['id']: state['position'] for state in states}
            for obj in scene['objects']:
                obj['position'] = list(positions[obj['id']])
        plan = plan_scene_motion(scene, prompt, start)
        return {'plan': plan, 'prior_positions': start} if plan is not None else None

    def _generation_prompt(self, prompt, context):
        return context['plan'].backend_prompt if context is not None else prompt

    def _generation_seconds(self, context):
        return getattr(context['plan'], 'recommended_seconds', None) if context is not None else None

    def _process_generation_result(self, result, context):
        if context is None:
            return result
        from scene_motion import apply_scene_motion
        result = apply_scene_motion(result, context['plan'], prior_positions=context['prior_positions'])
        context['prior_positions'] = result['positions'][-1].copy()
        return result

    def _invalidate_scene_motion(self):
        if getattr(self, 'busy', False):
            self._invalidate()
            self.status = 'Scene changed · generate again to use the updated geometry'

    @property
    def show_scene_targets(self):
        return getattr(self, '_show_scene_targets', False)

    @show_scene_targets.setter
    def show_scene_targets(self, visible):
        with self.lock:
            if self.show_scene_targets != bool(visible):
                self._show_scene_targets = bool(visible)
                # The viewer also redraws paused scenes on revision changes.
                self.project_revision += 1

    @staticmethod
    def _document(scene):
        doc = {'version': 3 if 'assets' in scene else 2,
               'name': scene.get('name', 'Untitled scene'), 'objects': scene.get('objects', []),
               'effects': scene.get('effects', []), 'lighting': scene.get('lighting', 'neutral')}
        if 'assets' in scene:
            doc['assets'] = scene['assets']
            for key in ('camera','targets'):
                if key in scene:doc[key]=scene[key]
        return validate_scene(doc)

    def scene_document(self):
        with self.lock:
            return self._document(self.scene)

    def set_scene(self, document):
        document = validate_scene(document)
        with self.lock:
            self._invalidate_scene_motion()
            self.scene.pop('assets', None)
            self.scene.pop('camera', None)
            self.scene.pop('targets', None)
            self.scene.update({k: copy.deepcopy(v) for k, v in document.items() if k != 'version'})
            if document['version'] == 3:
                self.scene['gate']['enabled'] = False
            self.project_revision += 1
            self._background_revision = getattr(self, '_background_revision', 0) + 1
            self.project_status = 'Unsaved scene changes · props, effects and lighting included'

    def generate_scene(self, prompt, generator=None, seed=0):
        prompt = validate_prompt(prompt)
        with self.lock:
            # Motion edits are independent of a background job. Retain identity
            # to reject project switches even when both projects have the same set.
            original_scene = self.scene
            original_document = self._document(self.scene)
            background_revision = getattr(self, '_background_revision', 0)
        doc = generator.generate(prompt) if generator else generate_recipe(prompt, seed)
        doc = validate_scene(doc)
        with self.lock:
            if (self.scene is not original_scene
                    or getattr(self, '_background_revision', 0) != background_revision
                    or self._document(self.scene) != original_document):
                raise ValueError('Project changed during generation; try again for the current background')
            self.set_scene(doc)
        return copy.deepcopy(doc)

    def edit_object(self, identifier, **changes):
        if set(changes) - {'position', 'size', 'color', 'yaw', 'interaction'}:
            raise ValueError('Unsupported prop edit')
        with self.lock:
            objects = copy.deepcopy(self.scene.get('objects', []))
            obj = next((o for o in objects if o['id'] == identifier), None)
            if obj is None:
                raise ValueError('Object no longer exists; refresh the list')
            obj.update(changes)
            self.set_objects(objects)

    def remove_object(self, identifier):
        with self.lock:
            self.set_objects([o for o in self.scene.get('objects', []) if o['id'] != identifier])

    def duplicate_object(self, identifier):
        with self.lock:
            objects = copy.deepcopy(self.scene.get('objects', []))
            original = next((o for o in objects if o['id'] == identifier), None)
            if original is None:
                raise ValueError('Object no longer exists; refresh the list')
            obj = copy.deepcopy(original)
            ids = {o['id'] for o in objects}
            i = 1
            while f"copy-{i}" in ids:
                i += 1
            obj['id'] = f'copy-{i}'
            obj['position'][0] += 1.2
            self.set_objects(objects + [obj])
            return obj['id']

    def set_objects(self, objects):
        objects = validate_objects(objects)
        with self.lock:
            candidate=dict(self.scene,objects=objects)
            if 'targets' in candidate:
                refs={o['id'] for o in objects}
                candidate['targets']=[t for t in candidate['targets'] if t['object_id'] in refs]
            self._document(candidate)
            self._invalidate_scene_motion()
            if 'targets' in candidate:self.scene['targets']=candidate['targets']
            self.scene['objects'] = objects
            self.project_revision += 1
            self._background_revision = getattr(self, '_background_revision', 0) + 1
            self.project_status = 'Unsaved object changes · Save project stores the scene'

    def generate_objects(self, prompt, generator=None):
        """Generate outside the lock; reject a result if project/scene changed meanwhile.

        UI callers should run this in a worker when using the gateway.
        """
        with self.lock:
            revision = self.project_revision
        objects = generator.generate(prompt) if generator is not None else generate_local(prompt)
        objects = validate_objects(objects)
        with self.lock:
            if revision != self.project_revision:
                raise ValueError('Project changed during object generation; regenerate for the current scene')
            self.set_objects(objects)
        return copy.deepcopy(objects)

    def load_objects(self, path):
        with open(path, 'rb') as source:
            raw = source.read(MAX_SCENE_FILE_BYTES + 1)
        if len(raw) > MAX_SCENE_FILE_BYTES:
            raise ValueError('Object file is too large')
        doc = json.loads(raw)
        self.load_scene_document(doc)

    def load_scene_document(self, doc):
        if isinstance(doc, dict) and set(doc) == {'version', 'objects'} and type(doc['version']) is int and doc['version'] == 1:
            self.set_scene({'version': 2, 'name': 'Imported objects', 'objects': doc['objects'], 'effects': [], 'lighting': 'neutral'})
        else:
            self.set_scene(doc)

    def load_project(self, data):
        # Validate before the base controller can replace any live state.
        _, _, _, scene = decode_project(data)
        self._document(scene)
        super().load_project(data)

    def object_states(self):
        with self.lock:
            # Recorded preview also demonstrates deterministic prop reactions.
            if isinstance(self.frame, bool) or not isinstance(self.frame, Integral) or not 0 <= self.frame < len(self.positions):
                raise ValueError('frame is outside positions')
            objects = self.scene.get('objects', [])
            if objects != getattr(self, '_state_objects_snapshot', None):
                canonical = validate_objects(objects)
                self._state_objects_snapshot = copy.deepcopy(objects)
                self._static_object_states = None
                if all(obj['interaction']['trigger'] == 'none' for obj in canonical):
                    static_states = [
                        {'id': obj['id'], 'position': obj['position'],
                         'color': obj['color'], 'active': False} for obj in canonical
                    ]
                    self._static_object_states = json.dumps(static_states)
            if self._static_object_states is None:
                states = evaluate_objects(objects, self.positions, self.frame, hand_indices=(25, 33))
            else:
                # Preserve per-call ownership of the public playback bundle.
                states = json.loads(self._static_object_states)
            assets = self.scene.get('assets', [])
            if assets != getattr(self, '_state_assets_snapshot', None):
                self._state_assets_snapshot = copy.deepcopy(assets)
                self._state_assets_json = json.dumps(assets)
            return {'objects': states,
                    'effects': copy.deepcopy(self.scene.get('effects', [])),
                    'seconds': self.frame / self.fps, 'lighting': self.scene.get('lighting', 'neutral'),
                    'assets': json.loads(self._state_assets_json),
                    'targets':copy.deepcopy(self.scene.get('targets',[])) if getattr(self,'show_scene_targets',False) else []}
