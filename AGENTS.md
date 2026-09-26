# Agent collaboration

Proactively use subagents for substantial tasks without waiting for the user to request them. Prefer bounded exploration, implementation, testing, and review; the main agent owns decisions, coordination, integration, and final verification.

Choose and explicitly set each subagent's available model and reasoning effort for task complexity and correctness risk. Use lighter models for straightforward, easily verified work and stronger models for ambiguous or difficult work. Escalate when verification shows the initial choice was insufficient. Parallelize independent work, avoid overlapping edits and duplicate effort, and handle trivial tasks directly. Briefly state each worker's assignment and requested model/effort.

## Shared machine coordination

The user has authorized coordination of the two existing workers and a forthcoming third worker. Coordination chat: `01a0dcd2-7034-78c2-b1ef-5dec9d37b278` ("Coordinate agents without blocking").

Before making changes for this coordinated workflow, read the shared coordination instructions and current ownership registry in:

`/Users/leoy/Code/ShellHacks/.runtime/agent-coordination/`

This absolute directory is shared across worktrees on this machine. Follow its README and publish/update your own worker claim with your chat ID, actual working directory, intended files, shared resources, and status. Only the coordinator edits the ownership registry; workers edit their own claims. Pass these instructions to your subagents. If the directory is not yet initialized, continue independent read-only work and report the missing registry in your chat.

- Continue independent work in parallel. Coordination is not a reason to stop all workers.
- Do not overwrite another worker's edits, switch its branch, stage its files, or stop its processes. Scope staging to your own reviewed changes.
- Use separate worktrees when overlapping code work needs isolation. Worktrees do not isolate ports, databases, shared virtual environments, remote files, GPU jobs, or services.
- Check ownership before editing an overlapping file or changing a shared resource. If it is already claimed or ownership is uncertain, record the dependency in your own claim and continue independent work until ownership is resolved.
- Coordinate shared live-app deployment, service restarts, SSH tunnels, package/environment changes, and RunPod operations. Identify the exact service/process/path and resource owner; do not use broad process-kill commands or unreviewed bulk file copies.
- Use worker-specific scratch paths and available dedicated ports. Release claims when the task completes; do not infer that an idle chat has released resources.
- A newly started third worker should register before editing. Registration does not authorize starting or expanding any implementation task beyond the user's request.

The registry is a cooperative protocol, not an operating-system lock. Recheck the relevant resource immediately before a shared mutation and resolve conflicting claims with its owner/coordinator.
