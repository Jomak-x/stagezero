"""Replaceable scene authoring controls, independent of the studio layout."""
import json
import threading
import math
from scene_composition import PRESETS, LIGHTING, make_preset
from scene_effects import EFFECT_KINDS, make_effect
from scene_generation import SceneGenerator, LocalSceneGenerator
from upload_events import install_upload_snapshots


def add_object_controls(gui, session):
    install_upload_snapshots(gui)
    with gui.add_folder('Scene generator', expand_by_default=True):
        prompt = gui.add_text('Describe a scene', initial_value='An enchanted grove with a glowing portal and fireflies', multiline=True)
        source = gui.add_dropdown('Generation source', ('Recipes · offline', 'AI gateway · Neon', 'Local AI · Ollama'))
        generate = gui.add_button('Generate scene')
        preset = gui.add_dropdown('Starter set', PRESETS)
        build = gui.add_button('Build starter set')
        frame_scene = gui.add_button('Frame whole scene')
        seed = gui.add_number('Variation seed', initial_value=0, min=0, max=1000000, step=1)
        status = gui.add_markdown('Recipes work immediately. AI composes catalog props and effects. Generation replaces the current set.')
        gui.add_markdown('Effects animate with playback and freeze when paused. Props react to motion; physical contact is not guaranteed.')
    with gui.add_folder('Effects & lighting', expand_by_default=False):
        lighting = gui.add_dropdown('Light palette', LIGHTING)
        apply_lighting = gui.add_button('Apply lighting')
        effect = gui.add_dropdown('Effect', tuple(EFFECT_KINDS))
        intensity = gui.add_slider('Effect density', min=0., max=1., step=.05, initial_value=.75)
        add_effect = gui.add_button('Add effect')
        clear_effects = gui.add_button('Clear effects')
    with gui.add_folder('Edit props', expand_by_default=False):
        selection = gui.add_dropdown('Prop', ('No props',))
        refresh = gui.add_button('Refresh prop list')
        coords = [gui.add_number(f'{axis} · m', initial_value=0., min=-20., max=20., step=.1) for axis in 'XYZ']
        move = gui.add_button('Move selected prop')
        duplicate = gui.add_button('Duplicate selected prop')
        remove = gui.add_button('Remove selected prop')
    with gui.add_folder('Scene files', expand_by_default=False):
        download = gui.add_button('Download scene JSON')
        upload = gui.add_upload_button('Import scene JSON', mime_type='.json')
        clear = gui.add_button('Clear scene')
    guard = threading.Lock()
    labels = {}

    def fit_scene(client):
        if client is None: return
        doc = session.scene_document()
        items = doc['objects'] + doc['effects']
        if not items: return
        lower = [min(o['position'][i] - o['size'][i] / 2 for o in items) for i in range(3)]
        upper = [max(o['position'][i] + o['size'][i] / 2 for o in items) for i in range(3)]
        center = [(lo + hi) / 2 for lo, hi in zip(lower, upper)]
        span = max(4., *(hi - lo for lo, hi in zip(lower, upper)))
        client.camera.position = (center[0] + span * .75, center[1] + span * .45, center[2] + span * 1.45)
        client.camera.look_at = tuple(center)
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = math.radians(45)

    @frame_scene.on_click
    def framed(event): fit_scene(event.client)

    def refresh_props():
        doc = session.scene_document()
        labels.clear()
        labels.update({f"{obj['name']} · {obj['id']}": obj['id'] for obj in doc['objects']})
        selection.options = tuple(labels) or ('No props',)
        if selection.value not in selection.options:
            selection.value = selection.options[0]
        lighting.value = doc['lighting']
        load_coords()

    def load_coords():
        identifier = labels.get(selection.value)
        doc = session.scene_document()
        obj = next((o for o in doc['objects'] if o['id'] == identifier), None)
        if obj:
            for handle, value in zip(coords, obj['position']): handle.value = value

    def report(doc):
        status.content = f"Scene ready · {len(doc['objects'])} props · {len(doc['effects'])} effects. Press Play to animate."
        refresh_props()

    def attempt(action):
        try:
            action()
            report(session.scene_document())
        except (ValueError, OSError) as exc:
            status.content = f'Could not apply change: {exc}'

    @generate.on_click
    def clicked(event):
        if not guard.acquire(blocking=False): return
        description, selected, variation = prompt.value, source.value, int(seed.value)
        generate.disabled = True
        status.content = 'Composing scene…'
        def run():
            try:
                generator = SceneGenerator() if selected == 'AI gateway · Neon' else LocalSceneGenerator() if selected == 'Local AI · Ollama' else None
                report(session.generate_scene(description, generator, variation))
                fit_scene(event.client)
            except ValueError as exc:
                status.content = f'Scene unchanged: {exc}'
            except Exception:
                status.content = 'Generation failed; current scene preserved.'
            finally:
                generate.disabled = False
                guard.release()
        threading.Thread(target=run, daemon=True).start()

    @build.on_click
    def build_clicked(event):
        attempt(lambda: session.set_scene(make_preset(preset.value, int(seed.value))))
        fit_scene(event.client)

    @apply_lighting.on_click
    def light_clicked(_):
        def action():
            doc = session.scene_document()
            doc['lighting'] = lighting.value
            session.set_scene(doc)
        attempt(action)

    @add_effect.on_click
    def effect_clicked(_):
        def action():
            doc = session.scene_document()
            ids = {e['id'] for e in doc['effects']}
            index = 0
            while f'{effect.value}-{index}' in ids: index += 1
            fx = make_effect(effect.value, index)
            fx['intensity'], fx['seed'] = intensity.value, int(seed.value)
            doc['effects'].append(fx)
            session.set_scene(doc)
        attempt(action)

    @clear_effects.on_click
    def effects_cleared(_):
        def action():
            doc = session.scene_document()
            doc['effects'] = []
            session.set_scene(doc)
        attempt(action)

    @selection.on_update
    def selected(_): load_coords()

    @refresh.on_click
    def refreshed(_): refresh_props()

    @move.on_click
    def moved(_):
        attempt(lambda: session.edit_object(labels.get(selection.value), position=[float(c.value) for c in coords]))

    @duplicate.on_click
    def duplicated(_): attempt(lambda: session.duplicate_object(labels.get(selection.value)))

    @remove.on_click
    def removed(_): attempt(lambda: session.remove_object(labels.get(selection.value)))

    @download.on_click
    def downloaded(event):
        if event.client:
            event.client.send_file_download('scene.stagezero.json', (json.dumps(session.scene_document(), indent=2)+'\n').encode())

    @upload.on_upload
    def uploaded(event):
        def action():
            data = event.file.content
            if len(data) > 100000: raise ValueError('Scene file exceeds 100 KB')
            session.load_scene_document(json.loads(data))
        attempt(action)

    @clear.on_click
    def cleared(_):
        attempt(lambda: session.set_scene({'version': 2, 'name': 'Empty scene', 'objects': [], 'effects': [], 'lighting': 'neutral'}))

    refresh_props()
