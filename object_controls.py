"""Replaceable scene authoring controls, independent of the studio layout."""
import json
import threading
import math
from scene_composition import PRESETS, LIGHTING, make_preset, encode_scene, MAX_SCENE_FILE_BYTES
from scene_effects import EFFECT_KINDS, make_effect
from adaptive_scene_generation import AdaptiveSceneGenerator
from object_generation import GatewayGenerator


def add_object_controls(gui, session):
    from scene_preset_previews import preset_preview

    browse = gui.add_button('Browse backgrounds', icon='photo')
    generate = gui.add_button('Generate background', icon='sparkles', color='teal')
    scene_view = gui.add_button('Best view', icon='camera', color='gray')
    frame_scene = gui.add_button('See whole background', icon='arrows-maximize', color='gray')
    status = gui.add_markdown('Choose a visual preset or describe a new background.')
    progress_bar = gui.add_progress_bar(0, visible=False, color='teal')
    with gui.add_folder('Customize background', expand_by_default=False):
        seed = gui.add_number('Variation', initial_value=0, min=0, max=1000000, step=1)
        with gui.add_folder('Cinematic effects & lighting', expand_by_default=False):
            lighting = gui.add_dropdown('Light palette', LIGHTING)
            apply_lighting = gui.add_button('Apply lighting')
            effect = gui.add_dropdown('Effect', tuple(EFFECT_KINDS))
            intensity = gui.add_slider('Effect intensity', min=0., max=1., step=.05, initial_value=.75)
            fx_position = gui.add_vector3('Effect position · m', initial_value=(0., 1.6, -2.5), step=.1)
            fx_scale = gui.add_slider('Effect scale', min=.25, max=2., step=.05, initial_value=1.)
            gui.add_markdown('Effects animate with Play. Place them behind the actor for a cinematic look.')
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
            target_info=gui.add_markdown('Show or focus attachment and landing markers.')
        with gui.add_folder('Scene files', expand_by_default=False):
            download = gui.add_button('Download scene JSON')
            upload = gui.add_upload_button('Import scene JSON', mime_type='.json')
            clear = gui.add_button('Clear scene')
    guard = threading.Lock()
    labels = {}
    target_labels = {}
    generation = {"percent": 0, "modal_progress": None, "modal_status": None}

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
        status.content = f"**{doc['name']}** · background ready"
        refresh_props()

    def attempt(action):
        try:
            action()
            report(session.scene_document())
        except (ValueError, OSError) as exc:
            status.content = f'Could not apply change: {exc}'

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
                    image = panel.add_image(preset_preview(name), label=name, visible=category == 'Cinematic')
                    choose = panel.add_button('Use ' + name, visible=category == 'Cinematic', color='teal')
                    cards.append((category, image, choose))

                    def apply_preset(selected_event, preset_name=name):
                        if not guard.acquire(blocking=False):
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

    def update_progress(percent, message):
        generation['percent'] = max(generation['percent'], percent)
        progress_bar.value = generation['percent']
        status.content = f"**{generation['percent']}% · {message}**\n\nEstimated progress · you can keep editing motion."
        # A user may dismiss the generation popup and keep working.
        for key, attribute, value in (
            ('modal_progress', 'value', generation['percent']),
            ('modal_status', 'content', status.content),
        ):
            handle = generation[key]
            if handle is not None:
                try:
                    setattr(handle, attribute, value)
                except (KeyError, RuntimeError):
                    generation[key] = None

    def provider_progress(message):
        lower = message.lower()
        milestones = [('designing', 15), ('refining generated', 35), ('saving', 50),
                      ('prepared', 58), ('composing', 65), ('refining the scene', 80),
                      ('composed', 90), ('ready', 95)]
        percent = next((value for word, value in milestones if word in lower), generation['percent'])
        friendly = ('Designing your background' if percent < 50 else
                    'Building the details' if percent < 65 else
                    'Arranging the scene' if percent < 90 else 'Finishing your background')
        update_progress(percent, friendly)

    @generate.on_click
    def generate_clicked(event):
        panel = event.client.gui if event.client is not None and hasattr(event.client, 'gui') else gui
        modal = panel.add_modal('Generate a background', size='md', show_close_button=True)
        with modal:
            prompt = panel.add_text('Describe your background', initial_value='', multiline=True,
                                    hint='For example: a neon rooftop above a rainy city')
            start = panel.add_button('Generate background', color='teal', icon='sparkles')
            modal_progress = panel.add_progress_bar(0, visible=False, color='teal')
            modal_status = panel.add_markdown('Describe the place and mood. We handle the details.')
            dismiss = panel.add_button('Close')

        @dismiss.on_click
        def dismiss_clicked(_):
            generation['modal_progress'] = generation['modal_status'] = None
            modal.close()

        @start.on_click
        def start_clicked(start_event):
            description = prompt.value.strip()
            if not description:
                modal_status.content = 'Describe a place first, such as “a cozy cafe at sunset”.'
                return
            if not guard.acquire(blocking=False):
                modal_status.content = 'A background is already being generated.'
                return
            variation = int(seed.value)
            generation.update(percent=0, modal_progress=modal_progress, modal_status=modal_status)
            generate.disabled = browse.disabled = start.disabled = True
            progress_bar.visible = modal_progress.visible = True
            progress_bar.animated = modal_progress.animated = True
            dismiss.label = 'Keep working in the editor'
            update_progress(5, 'Starting your background')

            def run():
                try:
                    try:
                        gateway = GatewayGenerator.from_env()
                    except ValueError:
                        gateway = None
                    generator = AdaptiveSceneGenerator(progress=provider_progress) if gateway else None
                    if generator is None:
                        update_progress(45, 'Building a matching preset layout')
                    document = session.generate_scene(description, generator, variation)
                    update_progress(100, 'Background ready')
                    report(document)
                    status.content = f"**{document['name']} is ready.**" + (' Built with a local preset layout.' if generator is None else '')
                    modal_status.content = status.content
                    view_scene(start_event.client)
                    dismiss.label = 'Done · view background'
                except ValueError as exc:
                    status.content = f'Background unchanged: {exc}'
                    modal_status.content = status.content
                except Exception:
                    status.content = 'Could not generate the background. Your current background is unchanged. Try a preset or try again.'
                    modal_status.content = status.content
                finally:
                    generate.disabled = browse.disabled = start.disabled = False
                    progress_bar.animated = modal_progress.animated = False
                    guard.release()
            threading.Thread(target=run, daemon=True).start()

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
            fx = make_effect(effect.value, index, position=list(fx_position.value))
            fx['size'] = [min(8., v * fx_scale.value) for v in fx['size']]
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
    def uploaded(_):
        def action():
            data = upload.value.content
            if len(data) > MAX_SCENE_FILE_BYTES: raise ValueError('Scene file exceeds 1 MB')
            session.load_scene_document(json.loads(data))
        attempt(action)

    @clear.on_click
    def cleared(_):
        attempt(lambda: session.set_scene({'version': 2, 'name': 'Empty scene', 'objects': [], 'effects': [], 'lighting': 'neutral'}))

    refresh_props()
