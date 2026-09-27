# Main UI cast integration — 2026-09-26

User approved integrating the reviewed multi-person milestone into main through PR #21. Read [MAIN-CAST.md](MAIN-CAST.md). The main studio keeps its layout; **Motion → Direct → AI cast · 1–3 people** adds the prompt workflow using the existing Scene tab, transport, timeline and save/open controls. The standalone prompt studio remains a research tool.

Preserve all original worker processes, private runtime files and rejected motion evidence. The authorized total RunPod ceiling is **$3/hour**; ask before exceeding it. Last observed existing running total was **$1.37/hour**, with no new Pod rented for this work. Check actual PR/runtime state before continuing; root owns final merge and deployment.

---

# Latest prompt-scene milestone — 2026-09-26

Work continues in `/Users/jakob/.codex/worktrees/motion-performance/Shellhacks`,
branch `codex/two-character-performance`, draft PR #21. Preserve the dirty original
workspace and other agents' running studios/workers. Main was fetched at `058d9a5`.

Read `review/prompt-scenes/README.md` and `review/prompt-scenes/review-results.json`
for the new standalone one-prompt, automatic-duration 1–3 actor testing UI,
fresh-model timings, full playback captures and rejected complex-action trials.
`prompt_scene_viewer.py` is separate from main's existing UI. Native Core/G1/paired
archives stay separate; cast scenes use `.cast.stagezero.npz`.

The warm native worker runs in its own process on the existing Pod, remote
loopback 8772 through local 8782. Private configuration is
`.runtime/prompt-native-provider.json`; do not commit its token or restart other
workers. No new Pod was rented. The normal user test link remains private Tailscale
port 2380; inspect process/registry state before changing it. Do not expose new
review routes without the outstanding user approval for that route.

The next boundary is user review. This is a tested research milestone, not general
contact or commercially cleared production animation. Three-person partner
changes work in the recorded scene; simultaneous three-person interaction and
dance/fall/help-up/hug remain unsolved. Preserve all failures and existing arch
platform provenance.

---

# Ready scene direction integration — 2026-09-26

The main studio now includes an additive **Motion → Scene direction · Core** panel.
Start with [SCENE-DIRECTION.md](SCENE-DIRECTION.md) for replay commands, measured live
results, limits, and the exact two-actor/city/arch archives in `review/studio-core/`.
The existing G1 studio, scene generation, objects and character workflows remain intact.
Core uses a separate native timeline and bundled human cast. Two actors have independent
prompts and shared playback; physical contact choreography remains experimental.
Known geometry enables object approach/open-arch routes; this is not image recognition.
The rejected swing/carry work and research-only model integrations are not promoted.
No extra Pod was rented and existing workers were preserved.

---

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
