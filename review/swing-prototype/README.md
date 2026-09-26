# Actual interactive run

[Watch the uncut 146-second browser recording](interactive-run.mp4).

The controls and HUD are part of the captured live canvas. This is one session:
start, two mid-swing turns, rooftop pickup, carried turn, landing, and ending.
It is not a collection of pre-rendered clips. The MP4 preserves the entire WebM
capture timeline; `manifest.json` records both hashes.

- [Measured results](../../docs/SWING-RESULTS.md)
- [Reproduction and integration](../../docs/SWING-PROTOTYPE.md)
- [Independent verification](verification.json)
- [Original recording report](recorded-report.json)
- [Carry close view](swing-carry-live.png)
- [Ending close view](swing-ending-live.png)

Raw logs, WebM, and seven actual native Core responses remain under the isolated
worktree's ignored `.runtime/swing/20260926-062105-a81b85/` directory. Run
`python experiments/verify_swing_run.py .runtime/swing/20260926-062105-a81b85`
to repeat the full recorded-data audit on this host.
