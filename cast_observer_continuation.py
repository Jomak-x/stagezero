"""Compose independently generated observer turns into terminal inactive spans.

Active pair tracks are immutable. Every attempted source/candidate is retained;
unsafe optional observer motion falls back to the existing authored observer.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from cast_motion_refinement import cast_body_clearance
from paired_meetup import _heading


def continue_released_observers(joints, actor_ids, segments, activities, *, client,
                                scene, folder, seed, scene_checker,
                                cancelled=lambda: False, source_paths=lambda: (),
                                on_record=lambda record: None):
    from cast_observer_turn import generate_observer_turn, ObserverTurnRejected

    out = np.asarray(joints).copy()
    activity = deepcopy(activities)
    records = []
    ids = list(actor_ids)
    for index, aid in enumerate(ids):
        mask = np.zeros(len(out), dtype=bool)
        for item in activity:
            if aid in item['active_actor_ids']:
                mask[item['start_frame']:item['end_frame_exclusive']] = True
        # Restrict this milestone to a released participant who will not reenter.
        last = np.flatnonzero(mask)
        if not len(last) or last[-1] == len(out)-1:
            continue
        start = int(last[-1])+1
        previous = next(s for s in segments if s['start_frame'] <= start-1 < s['end_frame_exclusive'])
        if start < 2 or previous['source'] != 'intergen' or len(out)-start < 150:
            continue
        following = next((item for segment, item in zip(segments, activity) if item['start_frame'] >= start
                          and segment['source'] == 'intergen'
                          and len(item.get('contact_actor_ids', [])) == 2
                          and aid not in item['contact_actor_ids']), None)
        if following is None:
            continue
        focus_ids = [ids.index(a) for a in following['contact_actor_ids']]
        lo, hi = following['start_frame'], following['end_frame_exclusive']
        target = np.median(out[lo:hi, focus_ids, 0], axis=(0, 1))[[0, 2]]
        delta = target-out[start-1, index, 0, [0, 2]]
        desired = math.atan2(delta[0], delta[1])
        angle = math.atan2(math.sin(desired-_heading(out[start-1, index])),
                           math.cos(desired-_heading(out[start-1, index])))
        if np.linalg.norm(delta) < .35 or abs(angle) < math.radians(35):
            continue
        if cancelled():
            raise RuntimeError('Observer continuation cancelled')
        name = f'observer-{aid}-{start}'
        path = Path(folder)/f'{name}-candidate.npz'
        record = {'actor_id': aid, 'start_frame': start, 'end_frame_exclusive': len(out),
                  'target_xz': target.tolist(), 'initial_heading_error_degrees': math.degrees(abs(angle)),
                  'status': 'generating', 'active_pair_frames_modified': False,
                  'source': 'independent ARDY Core observer turn; authored boundary bridge',
                  'visual_acceptance': 'unverified'}
        records.append(record)
        prior_sources = set(source_paths())
        candidate = None
        try:
            track, report = generate_observer_turn(
                out[start-2:start, index], client, actor_id=aid, target_xz=target,
                duration_frames=len(out)-start, seed=(seed+701+index) % 2**32,
                cancelled=cancelled)
            record['motion'] = report
            candidate = out.copy()
            candidate[start:, index] = track
            np.savez_compressed(path, joints=candidate)
            record['candidate_archive'] = str(path)
            # Check the actual moving cast, not static scene-idle planning boxes.
            before = cast_body_clearance(out, ids, activity)
            after = cast_body_clearance(candidate, ids, activity)
            clearance = []
            for pair, (new, unpaired) in after.items():
                if index not in pair:
                    continue
                old = before[pair][0]
                relevant = unpaired & (np.arange(len(out)) >= start)
                worsened = relevant & (new < 0) & (new < old-1e-4)
                clearance.append({'actor_ids': [ids[i] for i in pair],
                                  'minimum_clearance_m': float(new[relevant].min()) if relevant.any() else None,
                                  'overlap_frames': int(np.count_nonzero(relevant & (new < 0))),
                                  'worsened_overlap_frames': int(worsened.sum())})
            record['body_clearance'] = clearance
            if any(c['worsened_overlap_frames'] for c in clearance):
                raise ValueError('Observer turn worsens unintended body-proxy overlap')
            checks = []
            for segment, item in zip(segments, activity):
                if segment['end_frame_exclusive'] <= start:
                    continue
                checks.append(scene_checker(candidate[max(start-1, segment['start_frame']-1):segment['end_frame_exclusive']],
                                            scene, ids, contact_pair=item.get('contact_actor_ids', ())))
            record['scene_geometry'] = checks
            # Generated motion is excluded from the later authored upper-body
            # pass. Its stationary terminal tail can still breathe and settle.
            for span in report['spans']:
                if span['source'] == 'stationary_hold':
                    continue
                a, b = start+span['start_frame'], start+span['end_frame_exclusive']
                for item in activity:
                    left, right = max(a, item['start_frame']), min(b, item['end_frame_exclusive'])
                    if left < right:
                        item.setdefault('observer_control_spans', []).append({
                            'actor_id': aid, 'start_frame': left, 'end_frame_exclusive': right,
                            'source': span['source']})
            out = candidate
            record['status'] = 'accepted_by_mechanical_and_geometry_gates'
        except ObserverTurnRejected as exc:
            record['motion'] = exc.report
            partial = getattr(exc, 'candidate_joints', getattr(exc, 'candidate', None))
            if partial is not None:
                np.savez_compressed(path, joints=partial)
                record['candidate_archive'] = str(path)
                record['candidate_layout'] = 'observer-only generated candidate; may be partial'
            record.update(status='rejected_preserved_existing_observer', reason=str(exc))
        except ValueError as exc:
            record.update(status='rejected_preserved_existing_observer', reason=str(exc))
        except Exception as exc:
            record.update(status='failed_generation', reason=str(exc),
                          motion=getattr(exc, 'observer_report', {}))
            partial = getattr(exc, 'observer_candidate', None)
            if partial is not None:
                np.savez_compressed(path, joints=partial)
                record['candidate_archive'] = str(path)
                record['candidate_layout'] = 'observer-only generated candidate; may be partial'
            raise
        finally:
            record['source_archives'] = [p for p in source_paths() if p not in prior_sources]
            (Path(folder)/f'{name}-report.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
            on_record(record)
    return out, activity, records
