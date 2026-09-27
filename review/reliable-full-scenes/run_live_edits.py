"""Exercise live scene edits against a supplied project archive."""
import argparse
import copy
import json
import time
from pathlib import Path

def wait_until_idle(session, timeout):
    deadline = time.monotonic() + timeout
    while session.busy and time.monotonic() < deadline:
        time.sleep(0.05)
    return not session.busy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="StageZero project archive to edit")
    parser.add_argument("--out", default=".runtime/reliability-edits", help="directory for output projects and results.json")
    parser.add_argument("--token-path", default=".runtime/api-token", help="path to the local API token")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="live motion service URL")
    parser.add_argument("--edit-timeout", type=float, default=180, help="seconds to wait for each edit")
    parser.add_argument("--cancel-timeout", type=float, default=10, help="seconds to wait for cancellation generation calls")
    parser.add_argument("--cancel-only", action="store_true", help="skip the six edit cases and run only the cancellation check")
    args = parser.parse_args()

    import numpy as np
    from directing import DirectorSession
    from live_motion import Backend
    from scene_acceptance import analyze_take
    from takes import encode_project

    class CountBackend(Backend):
        def __init__(self, token_path, url):
            super().__init__(token_path, url)
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return super().generate(*args, **kwargs)

    raw = Path(args.archive).read_bytes()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = CountBackend(args.token_path, args.url)
    session = DirectorSession(
        backend,
        np.zeros((1, 34, 3), np.float32),
        np.tile(np.eye(3, dtype=np.float32), (1, 34, 1, 1)),
    )
    report = []
    failed = False

    if not args.cancel_only:
        for index, prompt in [
            (0, "A person walks forward confidently."),
            (6, "A person sidesteps right with broad steps."),
            (13, "A person walks backward slowly."),
        ]:
            for repetition in range(2):
                session.load_project(raw)
                original = session.takes[session.active_take]
                before = original.positions.copy()
                motion = original.motion.copy()
                segments = copy.deepcopy(original.segments)
                cut = segments[index]["start"]
                calls = backend.calls
                accepted = session.submit_action_edit(prompt, index, "replace", automatic_timing=True)
                idle = wait_until_idle(session, args.edit_timeout) if accepted else True
                take = session.takes[session.active_take]
                success = accepted and idle and take is not original
                row = {
                    "index": index,
                    "repeat": repetition + 1,
                    "success": success,
                    "status": session.status,
                    "calls": backend.calls - calls,
                    "prefix_preserved": bool(
                        np.array_equal(before[:cut], take.positions[:cut])
                        and np.array_equal(motion[:cut], take.motion[:cut])
                    ),
                }
                if success:
                    acceptance = analyze_take(take)
                    row["acceptance"] = acceptance
                    row["acceptance_passed"] = bool(acceptance.get("passed"))
                    project_path = out / f"edit-{index}-{repetition + 1}.stagezero.npz"
                    project_path.write_bytes(encode_project(session.takes, session.active_take, session.frame, session.scene))
                    undo_ok = bool(session.undo_action_edit())
                    restored = session.takes[session.active_take]
                    row["undo_exact"] = bool(
                        undo_ok
                        and np.array_equal(before, restored.positions)
                        and np.array_equal(motion, restored.motion)
                        and segments == restored.segments
                    )
                else:
                    row["original_preserved"] = bool(
                        take is original
                        and np.array_equal(before, take.positions)
                        and segments == take.segments
                    )
                    row["acceptance_passed"] = False
                    row["undo_exact"] = False
                report.append(row)
                (out / "results.json").write_text(json.dumps(report, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k != "acceptance"}), flush=True)
                failed |= not (
                    row["success"]
                    and row["prefix_preserved"]
                    and row["undo_exact"]
                    and row["acceptance_passed"]
                )

    session.load_project(raw)
    original = session.takes[session.active_take]
    calls = backend.calls
    accepted = session.submit_action_edit("A person walks forward.", 0, "replace", automatic_timing=True)
    deadline = time.monotonic() + args.cancel_timeout
    while backend.calls < calls + 2 and session.busy and time.monotonic() < deadline:
        time.sleep(0.02)
    cancellation_started = backend.calls >= calls + 2
    session.seek(session.frame)
    # Let any in-flight response arrive; cancellation must keep it from replacing the take.
    time.sleep(1.5)
    cancel_row = {
        "case": "cancel",
        "submitted": bool(accepted),
        "cancellation_started": cancellation_started,
        "original_preserved": session.takes[session.active_take] is original,
        "busy": session.busy,
    }
    report.append(cancel_row)
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(cancel_row), flush=True)
    failed |= not (
        cancel_row["submitted"]
        and cancel_row["cancellation_started"]
        and cancel_row["original_preserved"]
        and not cancel_row["busy"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
