# Create a character

Open **Character → Create from description**, describe one full-body person, and press **Create character**. The current actor stays visible while the appearance and 3D model are generated. A preview image, stage, and elapsed time show progress. The finished character joins the same saved list as imported GLBs and activates after the browser has loaded it.

Use a simple outfit and clearly separated arms and legs for the most reliable fit. Generation takes several minutes depending on the Gemini API and GPU load. Cancel stops this request; an image-provider request already in flight may finish. If you change characters during generation, your selection wins and the result remains saved for later.

The model has textured geometry and automatic body motion. Hands, fingers, and facial expressions remain static, and extreme poses can distort the mesh. This is an approximate generated actor, not a production animation rig.

## Existing private backend

The feature uses **Gemini 3.1 Flash-Lite** to turn your description into a short, detailed appearance brief. That brief drives the existing **Neon image generator**, and the private **TRELLIS GPU worker** reconstructs the textured 3D character. Gemini is used for text only; no Gemini image-generation quota is needed. All three stages must be configured.

- Install the normal `requirements-live.txt` dependencies and build `studio_client`.
- Put `GEMINI_API_KEY=your-key-here` in `.runtime/characters.env` at the project root (create `.runtime` if needed), or export `GEMINI_API_KEY` before starting the UI. The environment takes precedence. This file is Git-ignored; never put the key in frontend code. Get a key from [Google AI Studio](https://aistudio.google.com/apikey) with text-generation access and available quota. Restart the UI after changing its environment.
- Keep `NEON_AI_GATEWAY_BASE_URL` and `NEON_AI_GATEWAY_TOKEN` in the environment or private `.runtime/objects.env` file for image generation.
- Keep the worker bearer token in `.runtime/api-token`; SSH settings are in `.runtime/pod.env` (see `pod.env.example`). Never commit these files.
- `run-character-backend.command` connects to the already configured Pod and starts its existing isolated character environment. It does not provision GPU resources. That environment must already contain TRELLIS and its CUDA dependencies at `/workspace/stagezero-characters/TRELLIS`, with an isolated `.venv`, model access, and a verified smoke reference at `jobs/smoke/reference.png`.
- The worker listens privately through an SSH tunnel on `127.0.0.1:8770`. `STAGEZERO_CHARACTER_BACKEND_URL` can select another localhost tunnel. It admits one job at a time and waits for 14 GiB of free GPU memory; its PyTorch allocator is capped at 12 GiB.
- Start the main UI with `python director_viewer.py`. Characters persist under `.runtime/characters` (or `--characters PATH`).

The default Gemini text model is `gemini-3.1-flash-lite`. Optionally set `STAGEZERO_GEMINI_CHARACTER_DESIGN_MODEL` in the process environment. Neon image generation defaults to `gpt-6-astra`; `STAGEZERO_CHARACTER_IMAGE_MODEL` remains its override. The production path uses Gemini for the brief and does not silently substitute another provider if Gemini fails.

For a challenge submission, the accurate description is: “Gemini expands a user's character idea into an appearance brief, which guides Neon image generation and TRELLIS 3D reconstruction.”

## Verification

Run `python -m unittest discover -v` and, inside `studio_client`, `npm run build && npm test`. Coverage includes provider errors, cancellation, stale callbacks, browser-confirmed activation, saved catalog restoration, material preservation, and native GLB skinning parity with the fitted actor.
