> **2026-09-26 realtime integration:** See [REALTIME-RESULTS.md](REALTIME-RESULTS.md) and [REALTIME-DIRECTING.md](REALTIME-DIRECTING.md). Three full 600-frame scenes pass real GPU and independent artifact verification; warm Core 25/25 jobs pass. Private isolated viewer is on port 2350; the original studio remains separate. Paired InterGen is explicitly research-only. See the measured limits before making production claims.

> **2026-09-26 scene interaction milestone:** The latest isolated experiments and review URLs are documented in [SCENE-INTERACTION-RESULTS.md](SCENE-INTERACTION-RESULTS.md). Branch: `codex/scene-interaction-lab`. Native Core gates/contact, real AI planning and InterGen paired motion are verified research components; they have not replaced the live studio. Experimental GPU processes have exited. The earlier scope below is historical and was superseded by the user’s explicit request to explore interactions and other models.

# StageZero review handoff — 2026-09-26

Workspace: `/Users/jakob/Desktop/Shellhacks`; branch: `feat/directing-workflow`.
Repository: https://github.com/Jomak-x/stagezero.git.

The single-actor implementation has been preserved and completed for review.
Read `docs/DIRECTING.md` and `review/DIRECTING-RESULTS.md` for workflow, measured
results, screenshots and explicit motion-quality limitations. 24 offline tests
pass; the completed real Pod soak ran 144 requests with zero failures over
605.35 seconds. Tunnel failure/recovery and browser save/download/upload passed.

The existing private Tailscale URL on port 2334 proxies the director on port 2336.
Connection secrets remain in ignored `.runtime/pod.env`, `.runtime/api-token`
and `.runtime/known_hosts`. Use `./run-director.command` to recover; do not provision
or resize resources. The Pod was left running as requested by the prior scope.

Original private comparison project:
`.runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz`.
Additional review generations and automatic backups remain in `.runtime/projects`.
Original first 104 frames are exact across both endings. Wave peaks at 5.60 s;
overhead-arms ending does not achieve its requested gesture. Do not describe
semantic compliance as solved or motion as streamed.

Implementation: `directing.py`, `takes.py`, `director_viewer.py`, launch command;
regressions: `test_live_motion`, `test_directing`, `test_director_edges`.
Recorded and previous live viewers remain available. No glasses/multiple actors.

STOP for user review. Proposed next milestone: bounded motion-quality/steering
improvements on the same checkpoint, only after approval.
