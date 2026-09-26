# Create a character

Open **Character → Create from description**, describe one full-body person, and press **Create character**. The current actor stays visible while the appearance and 3D model are generated. A preview image, stage, and elapsed time show progress. The finished character joins the same saved list as imported GLBs and activates after the browser has loaded it.

Use a simple outfit and clearly separated arms and legs for the most reliable fit. Generation takes several minutes depending on the gateway and GPU load. Cancel stops this request; an image-provider request already in flight may finish. If you change characters during generation, your selection wins and the result remains saved for later.

The model has textured geometry and automatic body motion. Hands, fingers, and facial expressions remain static, and extreme poses can distort the mesh. This is an approximate generated actor, not a production animation rig.

## Existing private backend

The feature uses the configured Neon gateway to create a photographic reference, then a private TRELLIS worker to create its textured mesh. No Meshy account or key is used. Gateway access alone does not run the 3D model; the existing GPU worker is also required.

- Install the normal `requirements-live.txt` dependencies and build `studio_client`.
- Keep `NEON_AI_GATEWAY_BASE_URL` and `NEON_AI_GATEWAY_TOKEN` in the environment or private `.runtime/objects.env` file.
- Keep the worker bearer token in `.runtime/api-token`; SSH settings are in `.runtime/pod.env` (see `pod.env.example`). Never commit these files.
- `run-character-backend.command` connects to the already configured Pod and starts its existing isolated character environment. It does not provision GPU resources. That environment must already contain TRELLIS and its CUDA dependencies at `/workspace/stagezero-characters/TRELLIS`, with an isolated `.venv`, model access, and a verified smoke reference at `jobs/smoke/reference.png`.
- The worker listens privately through an SSH tunnel on `127.0.0.1:8770`. `STAGEZERO_CHARACTER_BACKEND_URL` can select another localhost tunnel. It admits one job at a time and waits for 14 GiB of free GPU memory; its PyTorch allocator is capped at 12 GiB.
- Start the main UI with `python director_viewer.py`. Characters persist under `.runtime/characters` (or `--characters PATH`).

Optional `STAGEZERO_CHARACTER_IMAGE_MODEL` and `STAGEZERO_CHARACTER_DESIGN_MODEL` overrides default to `gpt-6-astra`. The gateway must support image generation for the configured model.

## Verification

Run `python -m unittest discover -v` and, inside `studio_client`, `npm run build && npm test`. Coverage includes provider errors, cancellation, stale callbacks, browser-confirmed activation, saved catalog restoration, material preservation, and native GLB skinning parity with the fitted actor.
