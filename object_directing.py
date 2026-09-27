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

    def set_scene(self, document, *, reset_gate=True):
        document = validate_scene(document)
        with self.lock:
            self.scene.pop('assets', None)
            self.scene.pop('camera', None)
            self.scene.pop('targets', None)
            self.scene.update({k: copy.deepcopy(v) for k, v in document.items() if k != 'version'})
            if reset_gate and document['version'] == 3:
                self.scene['gate']['enabled'] = False
            self.project_revision += 1
            self.project_status = 'Unsaved scene changes · props, effects and lighting included'

    def generate_scene(self, prompt, generator=None, seed=0, *, expected_revision=None):
        prompt = validate_prompt(prompt)
        with self.lock:
            revision = self.project_revision
            if expected_revision is not None and expected_revision != revision:
                raise ValueError('Project changed before scene generation started; try again')
        doc = generator.generate(prompt) if generator else generate_recipe(prompt, seed)
        doc = validate_scene(doc)
        with self.lock:
            if revision != self.project_revision:
                raise ValueError('Project changed during generation; try again for the current scene')
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
            if 'targets' in candidate:self.scene['targets']=candidate['targets']
            self.scene['objects'] = objects
            self.project_revision += 1
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

    def append_catalog_object(self, prompt, *, expected_revision=None):
        """Add one matching offline catalog prop without replacing the scene."""
        with self.lock:
            revision = self.project_revision
            if expected_revision is not None and expected_revision != revision:
                raise ValueError('Project changed before object generation started; try again')
        generated = validate_objects(generate_local(prompt))
        if len(generated) != 1:
            raise ValueError('Describe exactly one catalog prop: door, lamp, ball or chair')
        with self.lock:
            if revision != self.project_revision:
                raise ValueError('Project changed during object generation; try again')
            existing = copy.deepcopy(self.scene.get('objects', []))
            ids = {obj['id'] for obj in existing}
            for obj in generated:
                base = obj['id']
                suffix = 1
                while obj['id'] in ids:
                    obj['id'] = f'{base}-{suffix}'
                    suffix += 1
                ids.add(obj['id'])
            self.set_objects(existing + generated)
            return copy.deepcopy(generated[0])

    def append_custom_object(self, prompt, generator, *, expected_revision=None):
        """Generate outside the lock, then append one asset and instance atomically."""
        validate_prompt(prompt)
        with self.lock:
            revision = self.project_revision
            if expected_revision is not None and expected_revision != revision:
                raise ValueError('Project changed before object generation started; try again')
        result = generator.generate(prompt)
        if not isinstance(result, dict) or set(result) != {'asset', 'object'}:
            raise ValueError('Generator must return one custom asset and object')
        with self.lock:
            if revision != self.project_revision:
                raise ValueError('Project changed during object generation; try again')
            document = self._document(self.scene)
            document['version'] = 3
            assets = document.setdefault('assets', [])
            asset = result['asset']
            existing = next((item for item in assets if item['id'] == asset.get('id')), None)
            if existing is None:
                assets.append(asset)
            elif existing != asset:
                raise ValueError('Generated asset ID conflicts with the current scene')
            obj = copy.deepcopy(result['object'])
            if obj.get('kind') != 'custom' or obj.get('asset') != asset.get('id'):
                raise ValueError('Generated object must reference its custom asset')
            ids = {item['id'] for item in document['objects']}
            base = obj.get('id')
            if not isinstance(base, str):
                raise ValueError('Generated object ID is invalid')
            suffix = 1
            while obj['id'] in ids:
                obj['id'] = f'{base[:60]}-{suffix}'
                suffix += 1
            document['objects'].append(obj)
            validated = validate_scene(document)
            # Preserve the current gate state and every scene setting. set_scene
            # intentionally resets the gate for a newly replaced v3 scene.
            self.scene.update({key: copy.deepcopy(value) for key, value in validated.items() if key != 'version'})
            self.project_revision += 1
            self.project_status = 'Unsaved object changes · Save project stores the scene'
            return copy.deepcopy(obj)

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
