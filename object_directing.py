"""UI-independent object editing integration for the directing controller."""
import copy
import json
from directing import DirectorSession
from takes import decode_project
from scene_objects import validate_objects, evaluate_objects
from object_generation import generate_local, validate_prompt
from scene_composition import validate_scene, generate_recipe
from scene_effects import validate_effects


class ObjectDirectorSession(DirectorSession):
    def scene_document(self):
        with self.lock:
            return validate_scene({'version': 2, 'name': self.scene.get('name', 'Untitled scene'),
                                   'objects': self.scene.get('objects', []), 'effects': self.scene.get('effects', []),
                                   'lighting': self.scene.get('lighting', 'neutral')})

    def set_scene(self, document):
        document = validate_scene(document)
        with self.lock:
            self.scene.update({k: copy.deepcopy(v) for k, v in document.items() if k != 'version'})
            self.project_revision += 1
            self.project_status = 'Unsaved scene changes · props, effects and lighting included'

    def generate_scene(self, prompt, generator=None, seed=0):
        prompt = validate_prompt(prompt)
        with self.lock:
            revision = self.project_revision
        doc = generator.generate(prompt) if generator else generate_recipe(prompt, seed)
        doc = validate_scene(doc)
        with self.lock:
            if revision != self.project_revision:
                raise ValueError('Project changed during generation; try again for the current scene')
            self.set_scene(doc)
        return copy.deepcopy(doc)

    def edit_object(self, identifier, **changes):
        if set(changes) - {'position', 'size', 'color'}:
            raise ValueError('Only position, size and color can be edited')
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

    def load_objects(self, path):
        with open(path, 'rb') as source:
            raw = source.read(100_001)
        if len(raw) > 100_000:
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
        validate_scene({'version': 2, 'name': scene.get('name', 'Untitled scene'),
                        'objects': scene.get('objects', []), 'effects': scene.get('effects', []),
                        'lighting': scene.get('lighting', 'neutral')})
        super().load_project(data)

    def object_states(self):
        with self.lock:
            # Recorded preview also demonstrates deterministic prop reactions.
            return {'objects': evaluate_objects(self.scene.get('objects', []), self.positions,
                                                self.frame, hand_indices=(25, 33)),
                    'effects': copy.deepcopy(self.scene.get('effects', [])),
                    'seconds': self.frame / self.fps, 'lighting': self.scene.get('lighting', 'neutral')}
