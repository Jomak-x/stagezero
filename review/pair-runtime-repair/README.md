# Main pair generation and export repair

Verified the user's exact prompt through the main Studio controls:

> Two people start at independent start points they walk towards each other and then start sparring each other but in the end they hug once they are done

The original generation rejected a sampled endpoint-velocity mismatch. `before-manifest.json` preserves the failure. The transition fallback now accounts for native source radial and angular velocity over the first/last five authored bridge frames, retaining the original duration, source frames, speed/seam thresholds and scene geometry gates. It runs only after the original bridge fails solely on endpoint velocity; excessive anatomical distortion is separately rejected.

Fresh actual Core + InterGen UI generation (seed 42) completed 573 frames / 19.1 seconds. Both characters approach, spar, and end in an embrace. The complete take was watched in Studio and exported successfully; `ui-export/playback.mp4`, its capture manifest and the playable archive preserve the result. This verifies this request end-to-end, not perfect contact or universal prompt success.

Playback readiness remains per connection and clip revision. A slow initial load gets a temporary loading notice rather than a false export/reload error after five seconds. Export can wait up to 30 seconds and resend the clip once to recover an absent acknowledgement. Late connect callbacks preserve valid acknowledgements for the same connection; disconnected/replaced tabs and revisions still fail capture.

Neon GPT-6 Astra reviewed the playback acknowledgement path. Codex GPT-6 Astra/high diagnosed and reproduced the transition failure from saved sources; the main agent integrated and verified the actual UI. Existing private provider configuration and the existing worker tunnel were restored; no new Pods or model architecture.

Current private Studio URL: http://jakobs-mac-mini.tail5a8376.ts.net:2370/ . Original processes remain available for rollback. No private credentials are included in this review folder.
