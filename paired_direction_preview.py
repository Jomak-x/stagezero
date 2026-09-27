"""Editable world-space start/meeting markers and read-only planned routes."""
from copy import deepcopy
import numpy as np


class PairedDirectionPreview:
    def __init__(self, server, on_change):
        self.server, self.on_change = server, on_change
        self.request = None
        self.handles, self.labels, self.lines = [], [], []
        self._syncing = False
        for i, name in enumerate(('First start', 'Second start', 'Meeting point')):
            h = server.scene.add_transform_controls(f'/direction-marks/{i}', scale=.45,
                active_axes=(True,False,True), disable_rotations=True,
                translation_limits=((-24,24),(0,0),(-24,24)), visible=False)
            self.handles.append(h)
            self.labels.append(server.scene.add_label(f'/direction-labels/{i}', name, visible=False))
            @h.on_update
            def changed(event, index=i):
                if self._syncing or self.request is None:
                    return
                p = self.handles[index].position
                field = self.request['meeting'] if index == 2 else self.request['starts'][index]
                field.update(x=round(float(p[0]),3), z=round(float(p[2]),3))
                if index == 2:
                    self.request['target_id'] = None
                self.labels[index].position = (p[0], .2, p[2])
                self.clear_routes()
                self.on_change(deepcopy(self.request))

    def clear_routes(self):
        for h in self.lines:
            h.remove()
        self.lines = []

    def hide(self):
        self.request = None
        self.clear_routes()
        for h in self.handles+self.labels:
            h.visible = False

    def show(self, request, plan=None):
        self._syncing = True
        try:
            self.request = deepcopy(request)
            self.clear_routes()
            for i, point in enumerate([*request['starts'], request['meeting']]):
                self.handles[i].position = (point['x'],0.,point['z'])
                self.labels[i].position = (point['x'],.2,point['z'])
                self.handles[i].visible = self.labels[i].visible = True
            if plan:
                routes = plan.get('routes', [])
                if isinstance(routes, dict):
                    routes = [routes[key] for key in request['actor_ids']]
                for i, route in enumerate(routes):
                    points = route.get('points', route.get('path', [])) if isinstance(route, dict) else route
                    p = np.asarray(points, dtype=float)
                    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
                        continue
                    world = np.column_stack((p[:,0], np.full(len(p), .04), p[:,1]))
                    self.lines.append(self.server.scene.add_line_segments(f'/direction-route/{i}',
                        points=np.stack((world[:-1], world[1:]), axis=1).astype(np.float32),
                        colors=(52,209,220) if i == 0 else (250,178,78), line_width=3))
        finally:
            self._syncing = False
