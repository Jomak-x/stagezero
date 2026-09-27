"""One-actor opt-in terrain/Core trial around the authored Koma station kiosk."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from scene_composition import validate_scene
from studio_interaction_scene import adapt_studio_scene
from core_spatial_commands import parse_commands, plan_command
from scene_interaction_geometry import SceneInteractionGeometry
from scene_navigation import plan_navigation_route

SOURCE = "studio_client/src/environments/EnvironmentScene.ts:663-672"
START = (-22., 0., 1.)
GOAL = (-22., 0., 7.)


def station_kiosk_scene():
    # Builder.local(-22, 4, +0.25 rad), with exact solid station() box parts.
    yaw = math.degrees(.25)
    c, s = math.cos(.25), math.sin(.25)
    boxes = [
        (0., 1.25, 0., 5., 2.5, 3.2),
        (0., 1.63, 1.64, 4.45, 1.15, .04),
        (0., 1.05, 1.9, 4.7, .12, .65),
        (0., 2.75, .2, 5.7, .18, 4.),
    ]
    # Asset bounds exactly enclose the union: X[-2.85,2.85], Y[0,2.84],
    # Z[-1.8,2.225]. Scene3 permits the thin fascia as a part of one asset.
    centre = np.array([0., 1.42, .2125])
    span = np.array([5.7, 2.84, 4.025])
    parts = [{"shape": "box", "position": ((np.array(row[:3])-centre)/span).tolist(),
              "size": (np.array(row[3:])/span).tolist(), "color": [154,159,147]}
             for row in boxes]
    kiosk = {"id": "koma-kiosk", "name": "Koma Express kiosk", "kind": "custom",
             "asset": "kiosk-boxes", "position": [-22.+s*centre[2], centre[1], 4.+c*centre[2]],
             "size": span.tolist(), "yaw": yaw, "color": [154,159,147],
             "interaction": {"action": "none", "trigger": "none", "radius": 0}}
    floor = {"id": "plaza-floor", "name": "Station plaza support", "kind": "custom",
             "asset": "box", "position": [-20., -.1, 4.], "size": [12., .2, 14.],
             "color": [154,159,147],
             "interaction": {"action": "none", "trigger": "none", "radius": 0}}
    return validate_scene({"version": 3, "name": "Hikari station Koma kiosk terrain route proxy",
        "lighting": "sunset", "effects": [],
        "assets": [
            {"id": "box", "name": "Rendered station box", "parts": [{
                "shape": "box", "position": [0,0,0], "size": [1,1,1], "color": [154,159,147]}]},
            {"id": "kiosk-boxes", "name": "Koma kiosk source boxes", "parts": parts},
        ], "objects": [floor,kiosk]})


def preflight():
    scene = station_kiosk_scene()
    geometry = SceneInteractionGeometry.from_scene(scene)
    direct = np.linspace(START, GOAL, 161)
    blocked = [i for i, (x, y, z) in enumerate(direct) if any(
        geometry.obstacle_at(x, y+o, z, radius=.18 if o == .4 else .60)
        for o in (.4, .9, 1.35))]
    adapted = adapt_studio_scene(scene)
    adapted["terrain_active"] = True
    action = parse_commands("walk 6m forward", adapted)[0]
    stages, route = plan_command(action, adapted, ("actor_1",), "actor_1", None,
        {"actor_1": {"position_xz": [START[0], START[2]], "yaw": 0.}})
    return scene, {"straight_sample_count": len(direct), "straight_blocked_count": len(blocked),
                   "straight_first_blocked_index": blocked[0] if blocked else None,
                   "route": route, "planned_horizons": route["schedule"]["frames"]//40}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--token-file", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8769")
    p.add_argument("--generate", action="store_true")
    args=p.parse_args()
    scene, evidence=preflight()
    print(json.dumps({"straight_blocked": evidence["straight_blocked_count"],
                      "planned_horizons": evidence["planned_horizons"],
                      "route_waypoints": evidence["route"]["waypoints"]}))
    if not args.generate:
        return
    if evidence["planned_horizons"] > 6:
        raise ValueError("Trial exceeds authorized six Core horizons")
    if args.token_file is None:
        raise ValueError("--token-file required for generation")
    from realtime_client import RealtimeClient
    from terrain_assisted_session import run_assisted_terrain_commands, save_native_terrain_result, save_assisted_result
    from grounded_character import _PARENTS
    class BoundedClient(RealtimeClient):
        submitted = 0
        def wait(self, body, *, cancelled=lambda: False):
            if self.submitted >= 6:
                raise ValueError("Six-horizon Core trial budget exhausted")
            # The existing worker predates optional terrain Y fields. This
            # route is strictly flat, so their values are the ordinary defaults.
            if any(abs(v) > 1e-8 for v in body.get("coordinate_frames_y", {}).values()):
                raise ValueError("Cannot omit a nonzero terrain frame origin")
            body = dict(body)
            body.pop("coordinate_frames_y", None)
            body["root_targets"] = {actor: [
                {key: value for key, value in target.items() if key != "root_height"}
                if abs(target.get("root_height", .95)-.95) < 1e-8 else
                (_ for _ in ()).throw(ValueError("Cannot omit nonneutral root height"))
                for target in targets]
                for actor, targets in body.get("root_targets", {}).items()}
            index = self.submitted
            self.submitted += 1
            def checkpoint(clip):
                directory = args.output / "native-horizons"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f"horizon-{index:02d}.npz",
                    positions=clip.positions, rotations=clip.rotations,
                    native_features=clip.native_features)
            return super().wait(body, on_chunk=checkpoint, cancelled=cancelled)
    client=BoundedClient(args.url,args.token_file.read_text().strip(), job_timeout=120)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"scene3.json").write_text(json.dumps(scene,indent=2)+"\n")
    (args.output/"preflight.json").write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n")
    start=time.monotonic()
    def on_native(result):
        save_native_terrain_result(result,args.output/"native.npz")
        (args.output/"native-ready.json").write_text(json.dumps({
            "elapsed_s":time.monotonic()-start,"native_frames":result.native_clip.frames,
            "native_features_shape":result.native_clip.native_features.shape,
            "route_measurements":result.measurements},indent=2,default=float)+"\n")
    try:
        result=run_assisted_terrain_commands(scene,"walk 6m forward",
            actor_ids=("actor_1",),actor_id="actor_1",
            initial_placements={"actor_1":{"position_xz":[START[0],START[2]],"yaw":0.}},
            client=client,on_native_ready=on_native,
            cancelled=lambda: time.monotonic()-start > 700)
    except Exception as exc:
        (args.output/"rejection.json").write_text(json.dumps({
            "error_type":type(exc).__name__,"reason":str(exc),"elapsed_s":time.monotonic()-start,
            "submitted_horizons":client.submitted},indent=2)+"\n")
        raise
    save_assisted_result(result,args.output/"assisted.npz")
    display=result.presentation.positions[0]
    payload={"fps":result.presentation.fps,"frames":display.round(6).tolist(),
             "parents":list(_PARENTS),"source":SOURCE,
             "coordinate_system":"world metres; XZ station location; Y up",
             "native_motion_source":"ARDY Core from existing authenticated 8769 service",
             "provenance":{"scene":"scene3.json","native":"native.npz","assisted":"assisted.npz"},
             "metrics":{"frame_count":len(display),"straight_blocked_count":evidence["straight_blocked_count"],
                        "planned_horizons":evidence["planned_horizons"],
                        "backend_flat_compatibility": "omitted zero Y origin and neutral 0.95 m root-height hints",
                        "elapsed_s":time.monotonic()-start,
                        "native_sha256":result.report["native_sha256"],
                        "report":result.report}}
    (args.output/"human17-world.json").write_text(json.dumps(payload,allow_nan=False)+"\n")
    print(json.dumps({"success":True,"frames":len(display),"elapsed_s":payload["metrics"]["elapsed_s"]}))


if __name__=="__main__": main()
