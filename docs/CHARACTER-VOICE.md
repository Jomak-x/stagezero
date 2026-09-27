# Voice control center and character lines

Turn on **Activate voice tools** in the inspector. **Direct motion** retains hold-to-speak (release submits), typed directions, queue status, and cancellation. Auto examples: `Generate a scene: walk then wave`, `Generate a short: bow`, `Edit Copper Finch action two to bow`. Exact, distinct take names are easiest to recognize.

Under **Character dialogue**, select a generated single-character take, pause/scrub to the start of the line, choose Female voices or Male voices and a voice, enter the text, then select **Generate line**. Press Play to hear it with the character motion and captions. Remove deletes the line; failed generation can be retried without regenerating motion. Voice tools can be turned off while already submitted jobs finish.

The account catalog currently supplies Bella and Sarah (female), Roger and Charlie (male). Available voices are fetched server-side, rather than assuming IDs that may be unavailable to the account. Configure `ELEVENLABS_API_KEY` on the server with voice-read and text-to-speech permissions. Microphone directions also need speech-to-text permission and localhost/HTTPS. No credentials are delivered to browsers.

Speech defaults to `eleven_flash_v2_5`; `ELEVENLABS_TTS_MODEL` overrides it. Generation runs in a bounded background queue independently of motion. Identical voice/text pairs reuse a bounded in-memory audio cache. Projects bundle the generated MP3s, so replay or reopening a saved project makes no speech request. The four measured short-line examples completed in 0.25–0.36 seconds; these are observations, not a latency guarantee.

A line that outlasts the take extends it by holding the final pose, with no GPU call. This is a frozen final pose, not generated idle motion. Captions and sound follow the take clock for pause, seek, speed changes, looping, and switching takes. Browser audio restrictions may require **Enable audio**; unavailable audio can be retried or played silently.

This version targets the G1 single-character take workflow. Native paired/cast modes disable line creation. It does not add lip sync, conversations, automatic dialogue extraction from scene prompts, or separate voices for simultaneous cast members. Those need integration with their distinct scene/playback model.

Saved older projects load without dialogue. Branches/trim retain the overlapping portion of cues; regeneration of an action removes its affected cues. Cancelled jobs and results from changed projects, takes, or modes cannot attach late audio. Saving enforces the combined project audio budget.

See [the real examples and verification](../review/character-voice/README.md).
