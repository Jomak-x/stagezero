# Full scenes from one prompt

Open **Full scene** in the Studio to describe a complete performance in its own popup. Leave **Length** on **Auto** to give each movement the time it needs, then generate. The planner estimates each action from its pace, distance, repetitions, and transitions; the total is the sum of those estimates. The popup shows the estimated scene length after planning and each movement’s duration in **Refine**. Estimates describe playback length, not generation wait time. The motion worker generates each movement using the preceding motion as context.

You can still choose a fixed length, including 60 seconds. The planner budgets different amounts of time for different actions instead of evenly dividing the scene. If the requested actions cannot fit the target, choose Auto or adjust the request; the app does not stretch a short plan or speed up a long one to force a fit. Timing is rounded to 25 fps. A scene supports up to 120 seconds and 16 movements, with up to 30 seconds per movement.

Auto gives a floor get-up a conservative estimate, then checks for a sustained upright pose before moving on. It can finish recovery early or generate up to two additional chunks within the existing duration limits. If the character still cannot get upright, the scene reports the failed movement instead of continuing into a dance while prone. Explicit timing is preserved. This is a pose-based recovery check, not a guarantee of stunt accuracy.

You can close the popup while a job runs and reopen it to check progress. Scene choices stay synchronized across connected tabs. Multiple scene requests queue without replacing the current take. Configured independent backend lanes can process separate scenes in parallel; movements within a scene always run in order.

Choose **Review scene** to see a completed scene, or **Edit movements** to refine it. Clicking a full-scene movement on the timeline opens the same popup editor. Describe the change and select **Update movement**; **Auto** handles movement timing, with a manual override under **Timing**. Optional length changes live under **Timing**; **Preview movement** closes the popup and plays it in the viewer. Earlier movements remain unchanged; the selected movement and following movements are regenerated together to maintain continuity. Undo restores the previous take after a completed edit. After reopening a saved project, **Refine** automatically uses the current take. Save the project using the normal Studio save control to retain the result.

The popup uses the current background and the G1 motion workflow. It does not independently stage a second actor or create background geometry. Spoken lines are retained in the plan with a warning; this workflow does not render speech audio. Requested stunts and object contact still depend on the motion engine.

## Configuration

Planning uses the existing private gateway configuration in `.runtime/objects.env` or the environment. `STAGEZERO_STORY_MODEL` optionally overrides the planner model. Motion uses the Studio's configured G1 backend and token. Credentials stay on the server.

`STAGEZERO_STORY_BACKENDS` can list comma-separated existing HTTP loopback backend URLs for independent queue lanes. Repeated URLs share one lane. Only list compatible G1 services; this setting does not provision Pods or establish tunnels.

The CLI also uses Auto by default; pass `--seconds 60` for a fixed target. The planner and queue enforce bounded durations, action counts, and pending jobs. Motion-quality rejections receive up to three attempts per chunk; connection and authentication errors are not retried as quality failures. Cancelled or failed jobs do not publish a partial take. Loading checks the original project/background to avoid attaching a result to an unrelated scene.
