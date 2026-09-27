"""Replaceable scene authoring controls, independent of the studio layout."""
import json
import threading
import math
from scene_composition import PRESETS, LIGHTING, make_preset, encode_scene, MAX_SCENE_FILE_BYTES
from scene_effects import EFFECT_KINDS, make_effect
from scene_generation import LocalSceneGenerator
from adaptive_scene_generation import AdaptiveSceneGenerator
from scene_asset_library import AssetLibrary
from object_generation import GatewayGenerator
from single_prop_generation import SinglePropGenerator
from bounded_upload import ScopedUploadLimits, acquire_scoped_upload_limits
from upload_events import install_upload_snapshots
from scene_preset_previews import preset_preview


def add_object_controls(gui, session):
    install_upload_snapshots(gui)
    try:
        GatewayGenerator.from_env()
        default_source = 'AI gateway · Neon'
    except ValueError:
        default_source = 'Recipes · offline'
    with gui.add_folder('Create a background', expand_by_default=True):
        browse = gui.add_button('Browse backgrounds', icon='photo')
        prompt = gui.add_text('Describe a scene', initial_value='A detailed city boulevard with varied storefronts, sidewalks and a layered skyline', multiline=True)
        source = gui.add_dropdown('Generation source', ('Recipes · offline', 'AI gateway · Neon', 'Local AI · Ollama'), initial_value=default_source)
        generate = gui.add_button('Generate background + scene · replace')
        with gui.add_folder('Advanced · reusable props', expand_by_default=False):
            prepare = gui.add_button('Prepare custom props · AI')
            reuse = gui.add_checkbox('Reuse saved prop library', initial_value=False)
        with gui.add_folder('Or start from a preset', expand_by_default=False):
            preset = gui.add_dropdown('Starter set', PRESETS)
            build = gui.add_button('Build starter set')
        with gui.add_folder('Camera and variation', expand_by_default=False):
            scene_view = gui.add_button('Scene camera')
            frame_scene = gui.add_button('Frame whole scene')
            seed = gui.add_number('Variation seed', initial_value=0, min=0, max=1000000, step=1)
        status = gui.add_markdown('Your new scene replaces the current background when it is ready.')
    with gui.add_folder('Add one object', expand_by_default=False):
        object_prompt = gui.add_text('Describe an object',
                                     initial_value='A brass telescope on a wooden tripod' if default_source == 'AI gateway · Neon' else 'A lamp',
                                     multiline=True)
        object_source = gui.add_dropdown('Object source', ('AI · custom geometry', 'Recipes · catalog props'),
                                         initial_value='AI · custom geometry' if default_source == 'AI gateway · Neon' else 'Recipes · catalog props')
        add_object = gui.add_button('Generate object · add to scene')
        object_status = gui.add_markdown('AI creates one original static 3D prop. Offline recipes add one functional door, lamp, ball or chair. Existing scene and objects stay in place.')
    with gui.add_folder('Effects & lighting', expand_by_default=False):
        lighting = gui.add_dropdown('Light palette', LIGHTING)
        apply_lighting = gui.add_button('Apply lighting')
        effect = gui.add_dropdown('Effect', tuple(EFFECT_KINDS))
        intensity = gui.add_slider('Effect density', min=0., max=1., step=.05, initial_value=.75)
        fx_position = gui.add_vector3('Effect position · m', initial_value=(0., 1.6, -2.5), step=.1)
        fx_scale = gui.add_slider('Effect scale', min=.25, max=2., step=.05, initial_value=1.)
        add_effect = gui.add_button('Add effect')
        clear_effects = gui.add_button('Clear effects')
    with gui.add_folder('Edit props', expand_by_default=False):
        selection = gui.add_dropdown('Prop', ('No props',))
        refresh = gui.add_button('Refresh prop list')
        coords = [gui.add_number(f'{axis} · m', initial_value=0., min=-100., max=100., step=.1) for axis in 'XYZ']
        dimensions = [gui.add_number(f'{axis} size · m', initial_value=1., min=.05, max=60., step=.1) for axis in 'XYZ']
        yaw = gui.add_number('Rotation · degrees', initial_value=0., min=-360., max=360., step=5.)
        move = gui.add_button('Apply prop transform')
        duplicate = gui.add_button('Duplicate selected prop')
        remove = gui.add_button('Remove selected prop')
    with gui.add_folder('Interaction targets', expand_by_default=False):
        show_targets=gui.add_checkbox('Show target markers',initial_value=False)
        target_selection=gui.add_dropdown('Target',('No targets',))
        focus_target=gui.add_button('Focus target')
        target_info=gui.add_markdown('Reusable attachment and landing points for planning demos. These markers do not simulate swinging or contact.')
    with gui.add_folder('Scene files', expand_by_default=False):
        download = gui.add_button('Download scene JSON')
        upload = gui.add_upload_button('Import scene JSON', mime_type='.json')
        clear = gui.add_button('Clear scene')
    # The production GuiApi has an owner; lightweight GUI fakes used by unit
    # tests do not implement Viser's transfer machinery.
    if hasattr(gui, '_owner'):
        scene_upload_limits = acquire_scoped_upload_limits(gui._owner, gui=gui,
                                                            factory=ScopedUploadLimits)
        scene_upload_limits.register(upload, max_bytes=MAX_SCENE_FILE_BYTES,
                                     on_error=lambda message: setattr(status, 'content', f'Scene import rejected: {message}'))
    guard = threading.Lock()
    labels = {}
    target_labels = {}
    prepared = {}

    def view_scene(client):
        camera = session.scene_document().get("camera")
        if client is not None and camera:
            client.camera.position = tuple(camera["position"])
            client.camera.look_at = tuple(camera["look_at"])
            client.camera.up_direction = (0, 1, 0)
            client.camera.fov = math.radians(45)
        else:
            fit_scene(client)

    @scene_view.on_click
    def viewed(event): view_scene(event.client)

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
        target_labels.clear()
        target_labels.update({f"{t['name']} · {t['id']}":t['id'] for t in doc.get('targets',[])})
        target_selection.options=tuple(target_labels) or ('No targets',)
        if target_selection.value not in target_selection.options:target_selection.value=target_selection.options[0]
        focus_target.disabled=not bool(target_labels)

    def load_coords():
        identifier = labels.get(selection.value)
        doc = session.scene_document()
        obj = next((o for o in doc['objects'] if o['id'] == identifier), None)
        if obj:
            for handle, value in zip(coords, obj['position']): handle.value = value
            for handle, value in zip(dimensions, obj['size']): handle.value = value
            yaw.value = obj.get('yaw', 0)
            yaw.disabled = False

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
        description, selected, variation, use_library = prompt.value, source.value, int(seed.value), bool(reuse.value)
        with session.lock:
            expected_revision = session.project_revision
        generate.disabled = prepare.disabled = add_object.disabled = True
        status.content = 'Starting scene generation…'
        def run():
            try:
                assets = (AssetLibrary().load_all(limit=16) if use_library else prepared.get(description)) if selected == 'AI gateway · Neon' else None
                if use_library and selected == 'AI gateway · Neon' and not assets:
                    raise ValueError('Saved library is empty; prepare custom props first')
                generator = AdaptiveSceneGenerator(prepared_assets=assets, progress=lambda text: setattr(status, 'content', text)) if selected == 'AI gateway · Neon' else LocalSceneGenerator() if selected == 'Local AI · Ollama' else None
                report(session.generate_scene(description, generator, variation, expected_revision=expected_revision))
                if isinstance(generator, AdaptiveSceneGenerator):
                    prepared.clear()
                    prepared[description] = generator.prepared_assets
                    status.content += f' Custom geometry checked with {generator._model_label(generator.gateway)}.'
                view_scene(event.client)
            except ValueError as exc:
                status.content = f'Scene unchanged: {exc}'
            except Exception:
                status.content = 'Generation failed; current scene preserved.'
            finally:
                generate.disabled = prepare.disabled = add_object.disabled = False
                guard.release()
        threading.Thread(target=run, daemon=True).start()

    @prepare.on_click
    def prepare_clicked(_):
        if not guard.acquire(blocking=False): return
        description = prompt.value
        generate.disabled = prepare.disabled = add_object.disabled = True
        def run():
            try:
                generator = AdaptiveSceneGenerator(progress=lambda text: setattr(status, 'content', text))
                prepared.clear()
                prepared[description] = generator.prepare_assets(description)
                source.value = 'AI gateway · Neon'
                status.content = f"Prepared {len(prepared[description])} custom props. Generate scene will use this pack."
            except Exception as exc:
                status.content = f'Could not prepare props: {exc}'
            finally:
                generate.disabled = prepare.disabled = add_object.disabled = False
                guard.release()
        threading.Thread(target=run, daemon=True).start()

    @add_object.on_click
    def object_clicked(_):
        if not guard.acquire(blocking=False): return
        description, selected = object_prompt.value, object_source.value
        with session.lock:
            expected_revision = session.project_revision
        generate.disabled = prepare.disabled = add_object.disabled = True
        object_status.content = 'Generating one object…'
        def run():
            try:
                if selected == 'AI · custom geometry':
                    added = session.append_custom_object(description, SinglePropGenerator(),
                                                         expected_revision=expected_revision)
                else:
                    added = session.append_catalog_object(description, expected_revision=expected_revision)
                refresh_props()
                label = next((label for label, identifier in labels.items() if identifier == added['id']), None)
                if label: selection.value = label
                object_status.content = 'Added one object to the current scene. Select it to adjust its position, size or rotation.'
            except ValueError as exc:
                object_status.content = f'Object not added: {exc}'
            except Exception:
                object_status.content = 'Object generation failed; current scene preserved.'
            finally:
                generate.disabled = prepare.disabled = add_object.disabled = False
                guard.release()
        threading.Thread(target=run, daemon=True).start()

    @build.on_click
    def build_clicked(event):
        attempt(lambda: session.set_scene(make_preset(preset.value, int(seed.value))))
        view_scene(event.client)

    @browse.on_click
    def browse_clicked(event):
        panel = event.client.gui if event.client is not None and hasattr(event.client, 'gui') else gui
        modal = panel.add_modal('Choose a background', size='xl', show_close_button=True)
        groups = {
            'Cinematic': PRESETS[:3],
            'Everyday': PRESETS[3:8],
            'Stylized': PRESETS[8:],
        }
        with modal:
            categories = panel.add_button_group('Collection', tuple(groups))
            cards = []
            for category, names in groups.items():
                for name in names:
                    visible = category == 'Cinematic'
                    image = panel.add_image(preset_preview(name), label=name, visible=visible)
                    choose = panel.add_button('Use ' + name, visible=visible, color='teal')
                    cards.append((category, image, choose))

                    def apply_preset(selected_event, preset_name=name):
                        if not guard.acquire(blocking=False):
                            status.content = 'Finish the current scene change before choosing a background.'
                            return
                        try:
                            session.set_scene(make_preset(preset_name, int(seed.value)))
                            report(session.scene_document())
                            view_scene(selected_event.client)
                            modal.close()
                        except (ValueError, OSError) as exc:
                            status.content = f'Could not apply background: {exc}'
                        finally:
                            guard.release()
                    choose.on_click(apply_preset)
            done = panel.add_button('Close')

        @categories.on_click
        def category_changed(_):
            for category, image, choose in cards:
                image.visible = choose.visible = category == categories.value

        @done.on_click
        def close_gallery(_):
            modal.close()

    @apply_lighting.on_click
    def light_clicked(_):
        def action():
            doc = session.scene_document()
            doc['lighting'] = lighting.value
            session.set_scene(doc, reset_gate=False)
        attempt(action)

    @add_effect.on_click
    def effect_clicked(_):
        def action():
            doc = session.scene_document()
            ids = {e['id'] for e in doc['effects']}
            index = 0
            while f'{effect.value}-{index}' in ids: index += 1
            fx = make_effect(effect.value, index, position=list(fx_position.value))
            fx['size'] = [min(8., size * fx_scale.value) for size in fx['size']]
            fx['intensity'], fx['seed'] = intensity.value, int(seed.value)
            doc['effects'].append(fx)
            session.set_scene(doc, reset_gate=False)
        attempt(action)

    @clear_effects.on_click
    def effects_cleared(_):
        def action():
            doc = session.scene_document()
            doc['effects'] = []
            session.set_scene(doc, reset_gate=False)
        attempt(action)

    @show_targets.on_update
    def target_markers_changed(_):
        session.show_scene_targets=bool(show_targets.value)

    @focus_target.on_click
    def target_focused(event):
        if event.client is None:return
        from scene_targets import resolve_targets
        doc=session.scene_document()
        target=next((t for t in resolve_targets(doc.get('targets',[]),doc['objects'])
                     if t['id']==target_labels.get(target_selection.value)),None)
        if target is None:return
        x,y,z=target['position']
        event.client.camera.position=(x+4,y+3,z+6)
        event.client.camera.look_at=(x,y,z)
        event.client.camera.up_direction=(0,1,0)
        show_targets.value=True
        session.show_scene_targets=True
        target_info.content=f"{target['kind'].replace('_',' ').capitalize()} · ({x:.2f}, {y:.2f}, {z:.2f}) m. Attached to {target['object_id']}."

    @selection.on_update
    def selected(_): load_coords()

    @refresh.on_click
    def refreshed(_): refresh_props()

    @move.on_click
    def moved(_):
        def action():
            changes = dict(position=[float(c.value) for c in coords], size=[float(c.value) for c in dimensions])
            if not yaw.disabled: changes['yaw'] = float(yaw.value)
            session.edit_object(labels.get(selection.value), **changes)
        attempt(action)

    @duplicate.on_click
    def duplicated(_): attempt(lambda: session.duplicate_object(labels.get(selection.value)))

    @remove.on_click
    def removed(_): attempt(lambda: session.remove_object(labels.get(selection.value)))

    @download.on_click
    def downloaded(event):
        if event.client:
            event.client.send_file_download('scene.stagezero.json', encode_scene(session.scene_document()))

    @upload.on_upload
    def uploaded(event):
        def action():
            data = event.file.content
            if len(data) > MAX_SCENE_FILE_BYTES: raise ValueError('Scene file exceeds 1 MB')
            session.load_scene_document(json.loads(data))
        attempt(action)

    @clear.on_click
    def cleared(_):
        attempt(lambda: session.set_scene({'version': 2, 'name': 'Empty scene', 'objects': [], 'effects': [], 'lighting': 'neutral'}))

    refresh_props()
