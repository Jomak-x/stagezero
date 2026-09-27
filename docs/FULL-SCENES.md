# Full scenes from one prompt

Open **Full scene** in the Studio to describe a complete performance in its own popup. Choose the total length (including 60 seconds), then generate. The planner turns the description into ordered movements; the motion worker generates each movement using the preceding motion as context.

You can close the popup while a job runs and reopen it to check progress. Multiple scene requests queue without replacing the current take. Configured independent backend lanes can process separate scenes in parallel; movements within a scene always run in order.

Load a completed scene to review it in the viewer. Open **Refine**, choose a movement, describe the change, and select **Update movement**. Optional length changes live under **Timing**; **Preview movement** closes the popup and plays it in the viewer. Earlier movements remain unchanged; the selected movement and following movements are regenerated together to maintain continuity. Undo restores the previous take after a completed edit. After reopening a saved project, **Refine** automatically uses the current take. Save the project using the normal Studio save control to retain the result.

The popup uses the current background and the G1 motion workflow. It does not independently stage a second actor or create background geometry. Spoken lines are retained in the plan with a warning; this workflow does not render speech audio. Requested stunts and object contact still depend on the motion engine.

## Configuration

Planning uses the existing private gateway configuration in `.runtime/objects.env` or the environment. `STAGEZERO_STORY_MODEL` optionally overrides the planner model. Motion uses the Studio's configured G1 backend and token. Credentials stay on the server.

`STAGEZERO_STORY_BACKENDS` can list comma-separated existing HTTP loopback backend URLs for independent queue lanes. Repeated URLs share one lane. Only list compatible G1 services; this setting does not provision Pods or establish tunnels.

The planner and queue enforce bounded durations, action counts, and pending jobs. Motion-quality rejections receive up to three attempts per chunk; connection and authentication errors are not retried as quality failures. Cancelled or failed jobs do not publish a partial take. Loading checks the original project/background to avoid attaching a result to an unrelated scene.
