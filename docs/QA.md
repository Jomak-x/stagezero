# Next validation pass

Prior results are historical measurements in `review/MILESTONE-2.md`, not a
claim that an offline Pod is currently serving inference.

## Without the Pod

- Run controller regression tests and compile Python sources.
- Check a clean clone resolves the exact ARDY submodule commit.
- Check credentials, dataset CSV, weights and environments are excluded.
- Verify onboarding explains private assets and environment configuration.
- Friend: implement the independent edge-case tests described in CONTRIBUTING.

## When the user restarts the existing Pod

1. Confirm current SSH host/port, update `.runtime/pod.env` if necessary, and
   check whether `/workspace/stagezero` and the checkpoint cache survived.
2. Restore the authenticated loopback backend and SSH tunnel. Record cold
   loading separately from warm generation; do not provision new resources.
3. On the actual MacBook, measure several fresh instructions through Tailscale.
   Check visible action changes, root travel, feet, camera and pose continuity.
4. Run a 10-minute generation/playback session; record failures, latency
   distribution, GPU memory and process memory. The prior soak was 100 seconds.
5. Interrupt/reconnect the tunnel; test rapid prompt changes, pause/reset while
   pending, recorded fallback, browser reload and backend restart.
6. Save results and screenshots, list limitations, then stop for review.

Acceptance: no stale motion appears after cancellation/reset; controls remain
responsive during generation/failure; recovered backend accepts new commands;
reported live motion is freshly generated; recorded and simulated cases remain
explicitly labeled. Motion quality still requires the user's visual review.
