"""Check queue cancellation and acceptance for a supplied scene plan."""
import argparse
import json
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON scene plan to submit three times")
    parser.add_argument("--out", default=".runtime/reliability-queue.json", help="path for results JSON")
    parser.add_argument("--token-path", default=".runtime/api-token", help="path to the local API token")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="live motion service URL")
    parser.add_argument("--timeout", type=float, default=180, help="seconds to wait for all jobs")
    args = parser.parse_args()

    from live_motion import Backend
    from scene_acceptance import analyze_take
    from story_jobs import StoryJobQueue

    plan = json.loads(Path(args.plan).read_text())
    queue = StoryJobQueue([Backend(args.token_path, args.url)])
    identifiers = [queue.submit(plan, automatic=True) for _ in range(3)]
    cancelled = queue.cancel(identifiers[1])
    print("CANCELLED_PENDING", cancelled, flush=True)
    results = []
    failed = False
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            states = [queue.snapshot(identifier) for identifier in identifiers]
            if all(state["status"] in ("completed", "failed", "cancelled") for state in states):
                break
            time.sleep(0.1)

        for position, (identifier, snapshot) in enumerate(zip(identifiers, states)):
            row = {"position": position, "status": snapshot["status"], "error": snapshot["error"]}
            if snapshot["status"] == "completed":
                acceptance = analyze_take(queue.result(identifier))
                row.update(
                    seconds=acceptance.get("seconds"),
                    movements=acceptance.get("segment_count"),
                    passed=bool(acceptance.get("passed")),
                    failed_checks=acceptance.get("failed_checks"),
                )
            results.append(row)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(json.dumps(results), flush=True)
        failed |= not cancelled
        failed |= results[0]["status"] != "completed"
        failed |= results[1]["status"] != "cancelled"
        failed |= results[2]["status"] != "completed"
        failed |= not results[0].get("passed", False)
        failed |= not results[2].get("passed", False)
        return 1 if failed else 0
    finally:
        for identifier in identifiers:
            queue.cancel(identifier)
        queue.close()


if __name__ == "__main__":
    raise SystemExit(main())
