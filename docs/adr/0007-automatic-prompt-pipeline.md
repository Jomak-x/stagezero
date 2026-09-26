# Automatic validated prompt pipeline

Status: Proposed
Date: 2026-09-26
Specification: user acceptance request, 2026-09-26, chat 01a0df25-eb3c-7102-a987-5ff4c79b01b2.

## Context

Studio previously treated prompt refinement and motion submission as independent
button actions. Clarifications could block generation without resuming it, while
validation exceptions collapsed schema errors and semantic changes into the same
warning. The warning was displayed in both explanation and status.

## Decision

A Generate intent owns an immutable editor/context snapshot through refinement,
optional clarification, validation and one motion submission. Clarification
answers resume that intent. Editing, cancelling or changing context invalidates
it; a stale response cannot replace the editor or submit motion. The original
and validated prompt remain inspectable. A failed refinement cannot fall through
to raw-prompt generation.

Core diagnostics retain the original prompt, answer context, raw provider output,
parsed document and exact failed validation rule. Schema/format problems receive
a bounded automatic repair using that feedback. Transport errors and genuine
semantic changes do not cause unbounded retries or silent fallback. Explicit
constraints remain enforced while equivalent wording is normalized.

Qualitative size words do not authorize invented numeric distance or duration.
Material jump height/travel ambiguity is resolved with a short question. Prompt
validation is separate from measured quality of generated motion.

## Verification boundary

Deterministic tests cover malformed responses, missing fields, semantic changes,
network failure, double clicks and stale responses. Private actual-app video
captures must show typing plus Generate and, separately, clarification plus
automatic continuation. Request logs must establish the exact prompt submitted.
Real text/motion inference and service lifecycle use the shared coordination
protocol; test doubles are labelled and do not establish real model success.
