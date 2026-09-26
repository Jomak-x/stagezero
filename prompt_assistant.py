"""Clarify motion directions before offering a bounded model-facing prompt.

The assistant is called only from an explicit user action. It never generates a
motion or changes the current take. A small local rule handles ambiguous turns
even when the optional Ollama service is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Callable, Mapping, Sequence

import requests


MAX_PROMPT_CHARS = 500
MAX_RESPONSE_BYTES = 32_768


@dataclass(frozen=True)
class PromptQuestion:
    id: str
    text: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptAssistantResult:
    refined_prompt: str
    explanation: str
    questions: tuple[PromptQuestion, ...]
    ready: bool
    source: str
    warning: str = ""


_TURN_BACK = re.compile(r"\bturn\s+back\b|\bturn\s+around\b|\b(?:geri|geriye|arkana|arkaya)\s+d[oö]n\b", re.I)
_EXPLICIT_OPPOSITE = re.compile(
    r"\bhalf[- ]turn\b|"
    r"\bopposite\s+direction\b|\bface\s+the\s+opposite\b|"
    r"\bturn\s+around\b|\bters\s+y[oö]ne\b|\barkaya\s+bakarak\b", re.I,
)
_EXPLICIT_180 = re.compile(r"\b180\s*(?:°|degrees?|derece)(?=\W|$)|\b180\s+in place\b", re.I)
_CURRENT_HEADING_PREFIX = re.compile(
    r"\b(?:current(?:ly)?|şu an|şu anda|şimdi)\s*(?:facing|heading(?: is)?|oriented|at|bakıyor|dönük)?\s*$", re.I,
)
_EXPLICIT_RETURN = re.compile(
    r"\b(?:return|go)\s+to\s+(?:the\s+)?(?:start|starting\s+point|origin|previous\s+(?:heading|position))\b|"
    r"\bba[sş]lang[ıi][cç]\s+(?:noktas[ıi]na|konumuna)\b", re.I,
)
_EXPLICIT_BACKWARD = re.compile(r"\b(?:walk|step|move)\s+backward\b|\bgeri\s+geri\s+y[uü]r[uü]\b", re.I)
_DEICTIC = re.compile(r"\b(?:over there|like before|same as before|that direction|oraya|[oö]nceki gibi)\b", re.I)
_BODY_TURN = re.compile(r"\bturn(?:s)?\b(?!\s+(?:off|on|up|down|the)\b)", re.I)
_ANGLE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:°|degrees?|derece)(?=\W|$)", re.I)
_SIDE_ANGLE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:left|right)\b", re.I)
_TURKISH_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TURKISH_WORDS = re.compile(r"\b(?:geri|geriye|d[oö]n|sağa|sola|yerinde|yavaşça|hızlıca|şimdi|su an)\b", re.I)
_QUESTION_ID = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_UNCERTAIN = re.compile(r"^(?:yes|no|maybe|unsure|i don't know|evet|hayır|bilmiyorum|emin değilim)[.!? ]*$", re.I)


def _turkish(prompt: str) -> bool:
    return bool(_TURKISH_CHARS.search(prompt) or _TURKISH_WORDS.search(prompt))


def _question(prompt: str, key: str) -> PromptQuestion:
    tr = _turkish(prompt)
    if key == "turn_back_meaning":
        if tr:
            return PromptQuestion(key, "Geri dön derken hangi hareketi kastediyorsunuz?", (
                "Yerinde 180° dönüp ters yöne bak", "Önceki bakış yönüne dön",
                "Önceki konuma dön", "Yönünü değiştirmeden geriye yürü",
            ))
        return PromptQuestion(key, "What should “turn back” mean here?", (
            "Turn 180° in place to face the opposite direction",
            "Return to the previous heading", "Return to the previous position",
            "Walk backward without turning",
        ))
    if key == "turn_direction":
        return PromptQuestion(key,
            "Hangi yöne ve ne kadar dönsün?" if tr else "Which direction and how far should the person turn?",
            ("Sola / left", "Sağa / right", "180° / opposite direction"))
    return PromptQuestion(key,
        "Hangi konumu veya hareketi kastediyorsunuz?" if tr else "What position or motion does that refer to?")


def _answer(answers: Mapping[str, str], key: str) -> str:
    value = answers.get(key, "")
    if not isinstance(value, str):
        raise ValueError("Assistant answers must be text")
    value = " ".join(value.split())
    if len(value) > 300:
        raise ValueError("Assistant answer is too long")
    return value


def _turn_back_interpretation(answer: str) -> str:
    if not answer or _UNCERTAIN.fullmatch(answer):
        return ""
    lower = answer.casefold()
    if re.search(r"180|opposite|ters y[oö]n", lower):
        return "opposite"
    if re.search(r"previous heading|bak[ıi][sş] y[oö]n", lower):
        return "heading"
    if re.search(r"previous position|[oö]nceki konum|starting point|origin|ba[sş]lang[ıi][cç]", lower):
        return "position"
    if re.search(r"backward|geri(?:ye)? y[uü]r[uü]|geri geri", lower):
        return "backward"
    return "custom" if len(answer) >= 8 else ""


def _has_action_180(prompt: str) -> bool:
    for match in _EXPLICIT_180.finditer(prompt):
        if not _CURRENT_HEADING_PREFIX.search(prompt[max(0, match.start() - 45):match.start()]):
            return True
    return False


def _translate_turkish_details(details: str) -> str:
    """Translate common orientation/speed modifiers; retain unfamiliar text."""
    clauses = []
    angle_pattern = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:°|derece)?\s*(sağa|sola)\b", re.I)
    angles = list(angle_pattern.finditer(details))
    for index, angle_side in enumerate(angles):
        start, end = angle_side.span()
        previous_end = angles[index - 1].end() if index else 0
        next_start = angles[index + 1].start() if index + 1 < len(angles) else len(details)
        left_bound = max(previous_end, details.rfind(",", 0, start) + 1,
                         details.rfind(";", 0, start) + 1)
        punctuation = [position for mark in ",;" if (position := details.find(mark, end)) >= 0]
        right_bound = min([next_start, *punctuation])
        before = details[left_bound:start]
        after = details[end:right_bound]
        is_current = bool(re.search(r"\bmevcut\s+yön\b", before, re.I)
                          or re.search(r"\b(?:bakıyor|bakarken|dönük|yönelmiş)\b", after, re.I))
        label = "Current facing direction" if is_current else "Turn direction"
        side = "right" if angle_side.group(2).casefold() == "sağa" else "left"
        clauses.append(f"{label}: {angle_side.group(1).replace(',', '.')}° to the {side}.")
    details = angle_pattern.sub(" ", details)
    for word, clause in (("yavaşça", "Move slowly."), ("hızlıca", "Move quickly.")):
        if re.search(rf"\b{word}\b", details, re.I):
            clauses.append(clause)
            details = re.sub(rf"\b{word}\b", " ", details, flags=re.I)
    details = re.sub(r"\b180\s*(?:°|derece)(?=\W|$)", " ", details, flags=re.I)
    details = re.sub(r"\b(?:şu an|şu anda|şimdi|bakıyor|bakarken|dönük|yerinde|ve|sonra|karakter|kişi)\b",
                     " ", details, flags=re.I).strip(" ,.;")
    if details:
        clauses.append(f"Additional original details (Turkish): {details}.")
    return " ".join(clauses)


def _pending_question(prompt: str, answers: Mapping[str, str]) -> PromptQuestion | None:
    if _TURN_BACK.search(prompt) and not (
        _EXPLICIT_OPPOSITE.search(prompt) or _has_action_180(prompt)
        or _EXPLICIT_RETURN.search(prompt) or _EXPLICIT_BACKWARD.search(prompt)
    ) and not _turn_back_interpretation(_answer(answers, "turn_back_meaning")):
        return _question(prompt, "turn_back_meaning")
    if _DEICTIC.search(prompt) and (not _answer(answers, "reference")
                                    or _UNCERTAIN.fullmatch(_answer(answers, "reference"))):
        return _question(prompt, "reference")
    if (_BODY_TURN.search(prompt) and not _TURN_BACK.search(prompt)
            and not re.search(r"\b(?:left|right|clockwise|counterclockwise|opposite|around|previous heading)\b|\d+\s*(?:°|degrees?)", prompt, re.I)
            and (not _answer(answers, "turn_direction")
                 or _UNCERTAIN.fullmatch(_answer(answers, "turn_direction")))):
        return _question(prompt, "turn_direction")
    return None


def needs_clarification(prompt: str) -> bool:
    """Cheap no-network guard for unresolved motion directions."""
    if not isinstance(prompt, str):
        return False
    return _pending_question(prompt, {}) is not None


def _validate_inputs(prompt: str, answers: Mapping[str, str] | None) -> tuple[str, dict[str, str]]:
    if not isinstance(prompt, str):
        raise ValueError("Motion direction must be text")
    prompt = " ".join(prompt.split())
    if not prompt:
        raise ValueError("Enter a motion direction first")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Motion direction must be at most {MAX_PROMPT_CHARS} characters")
    if answers is None:
        return prompt, {}
    if not isinstance(answers, Mapping) or len(answers) > 8:
        raise ValueError("Assistant answers are invalid")
    normalized = {}
    for key in answers:
        if not isinstance(key, str) or not _QUESTION_ID.fullmatch(key):
            raise ValueError("Assistant answer key is invalid")
        normalized[key] = _answer(answers, key)
    return prompt, normalized


def _offline_prompt(prompt: str, answers: Mapping[str, str]) -> str:
    """Keep all user constraints while making a resolved turn unambiguous."""
    meaning = _turn_back_interpretation(_answer(answers, "turn_back_meaning"))
    if not meaning and _turkish(prompt) and _TURN_BACK.search(prompt) and (
        _EXPLICIT_OPPOSITE.search(prompt) or _has_action_180(prompt)
    ):
        meaning = "opposite"
    if _TURN_BACK.search(prompt) and meaning:
        replacements = {
            "opposite": "turn 180° in place to face the opposite direction",
            "heading": "turn to the previous heading",
            "position": "return to the previous position",
            "backward": "walk backward without turning",
        }
        if meaning == "custom":
            replacement = f"perform this motion: {_answer(answers, 'turn_back_meaning')}"
        else:
            replacement = replacements[meaning]
        selected_option = _answer(answers, "turn_back_meaning") in _question(prompt, "turn_back_meaning").options
        if _turkish(prompt):
            # Known Turkish turn phrases can be translated without losing a
            # stated angle or side. Keep extra original details visibly quoted.
            details = _TURN_BACK.sub("", prompt).strip(" ,.;")
            if details:
                prompt = f"A person should {replacement}. {_translate_turkish_details(details)}"
            else:
                prompt = f"A person should {replacement}."
        else:
            prompt = _TURN_BACK.sub(replacement, prompt, count=1)
        if not selected_option and meaning != "custom":
            answer = _answer(answers, "turn_back_meaning")
            if answer:
                prompt = f"{prompt.rstrip('. ')}. Clarification: {answer}."
    elif _turkish(prompt):
        prompt = f'Animate the motion described in this Turkish direction, preserving every detail: "{prompt}"'
    reference = _answer(answers, "reference")
    if reference:
        prompt = _DEICTIC.sub(lambda match: f"to {reference}" if match.group().casefold() == "over there" else reference,
                              prompt)
    extras = []
    for key, value in answers.items():
        if key == "turn_back_meaning" or not value:
            continue
        if key == "turn_direction":
            extras.append(f"Turn direction and extent: {value}")
        elif key == "reference":
            continue
        else:
            extras.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    if extras:
        prompt = f"{prompt.rstrip('. ')}. " + ". ".join(extras) + "."
    return prompt


def _offline_result(prompt: str, answers: Mapping[str, str], warning: str = "") -> PromptAssistantResult:
    pending = _pending_question(prompt, answers)
    tr = _turkish(prompt)
    if pending:
        return PromptAssistantResult("", "Önce bu ayrıntıyı netleştirelim." if tr else
                                     "Please clarify this detail before generating motion.",
                                     (pending,), False, "offline", warning)
    refined = _offline_prompt(prompt, answers)
    if len(refined) > MAX_PROMPT_CHARS:
        question = PromptQuestion("shorten", "Lütfen isteği kısaltın." if tr else
                                  "Please shorten the direction so all details fit.")
        return PromptAssistantResult("", "Ayrıntılar korunamadı." if tr else
                                     "The details could not fit without removing some of them.",
                                     (question,), False, "offline", warning)
    explanation = ("Yerel rehberlik: verdiğiniz ayrıntılar korundu."
                   if tr else "Offline guidance preserved the details you provided.")
    return PromptAssistantResult(refined, explanation, (), True, "offline", warning)


def _schema() -> dict:
    question = {"type": "object", "properties": {
        "id": {"type": "string", "maxLength": 40},
        "text": {"type": "string", "maxLength": 240},
        "options": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 5},
    }, "required": ["id", "text", "options"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "ready": {"type": "boolean"},
        "refined_prompt": {"type": "string", "maxLength": MAX_PROMPT_CHARS},
        "explanation": {"type": "string", "maxLength": 400},
        "questions": {"type": "array", "items": question, "maxItems": 3},
    }, "required": ["ready", "refined_prompt", "explanation", "questions"],
        "additionalProperties": False}


class OllamaPromptProvider:
    """Local structured-output provider; no model install or service startup."""

    def __init__(self, model: str | None = None, transport=None):
        self.model = model or os.environ.get("STAGEZERO_LOCAL_MODEL", "qwen3:4b")
        self.transport = transport or requests

    def __call__(self, system: str, user: str) -> Mapping[str, object]:
        with self.transport.post("http://127.0.0.1:11434/api/chat", json={
            "model": self.model, "stream": False, "think": False, "format": _schema(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.2, "num_predict": 850},
        }, timeout=(3, 55), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError("Local prompt model unavailable")
            body = bytearray()
            for chunk in response.iter_content(8192):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("Local prompt model response is too large")
        envelope = json.loads(body)
        return json.loads(envelope["message"]["content"])


def _validate_model_result(doc: Mapping[str, object], source: str) -> PromptAssistantResult:
    if not isinstance(doc, Mapping) or set(doc) != {"ready", "refined_prompt", "explanation", "questions"}:
        raise ValueError("Invalid assistant response")
    ready, refined, explanation, raw_questions = (doc[key] for key in
                                                   ("ready", "refined_prompt", "explanation", "questions"))
    if type(ready) is not bool or not isinstance(refined, str) or not isinstance(explanation, str):
        raise ValueError("Invalid assistant response fields")
    refined = " ".join(refined.split())
    explanation = " ".join(explanation.split())
    if len(refined) > MAX_PROMPT_CHARS or len(explanation) > 400 or not explanation:
        raise ValueError("Invalid assistant response length")
    if not isinstance(raw_questions, list) or len(raw_questions) > 3:
        raise ValueError("Invalid assistant questions")
    questions = []
    seen = set()
    for item in raw_questions:
        if not isinstance(item, Mapping) or set(item) != {"id", "text", "options"}:
            raise ValueError("Invalid assistant question")
        key, text, options = item["id"], item["text"], item["options"]
        if (not isinstance(key, str) or not _QUESTION_ID.fullmatch(key) or key in seen
                or not isinstance(text, str) or not text.strip() or len(text) > 240
                or not isinstance(options, list) or len(options) > 5
                or any(not isinstance(option, str) or not option.strip() or len(option) > 120
                       for option in options)):
            raise ValueError("Invalid assistant question fields")
        seen.add(key)
        questions.append(PromptQuestion(key, text.strip(), tuple(option.strip() for option in options)))
    if (ready and (not refined or questions)) or (not ready and (refined or not questions)):
        raise ValueError("Inconsistent assistant readiness")
    return PromptAssistantResult(refined, explanation, tuple(questions), ready, source)


def _angle_values(text: str) -> set[str]:
    return {match.group(1) for pattern in (_ANGLE, _SIDE_ANGLE) for match in pattern.finditer(text)}


def _sides(text: str) -> set[str]:
    found = set()
    if re.search(r"(?<![a-z])left\b|\bsola\b", text, re.I):
        found.add("left")
    if re.search(r"(?<![a-z])right\b|\bsağa\b", text, re.I):
        found.add("right")
    return found


def _validate_constraints(prompt: str, answers: Mapping[str, str], refined: str) -> None:
    """Reject a model answer that drops or invents explicit angles/sides."""
    source = " ".join((prompt, *answers.values()))
    required_angles = _angle_values(source)
    result_angles = _angle_values(refined)
    if _turn_back_interpretation(_answer(answers, "turn_back_meaning")) == "opposite":
        required_angles.add("180")
    allowed_angles = required_angles
    if not required_angles.issubset(result_angles) or not result_angles.issubset(allowed_angles):
        raise ValueError("Assistant changed an explicit angle")
    if _sides(source) != _sides(refined):
        raise ValueError("Assistant changed an explicit side")
    if (re.search(r"\bin place\b|\byerinde\b", source, re.I)
            and not re.search(r"\bin place\b", refined, re.I)):
        raise ValueError("Assistant dropped an in-place constraint")


def refine_prompt(
    prompt: str,
    answers: Mapping[str, str] | None = None,
    *,
    provider: Callable[[str, str], Mapping[str, object]] | None = None,
    offline: bool = False,
    model: str | None = None,
    history: Sequence[Mapping[str, str]] | None = None,
) -> PromptAssistantResult:
    """Ask at most three questions or return a <=500-character English direction.

    Answers are keyed by question ID and should be retained across rounds.
    ``provider`` is injectable for UI tests; ``offline=True`` avoids I/O.
    """
    prompt, answers = _validate_inputs(prompt, answers)
    pending = _pending_question(prompt, answers)
    if pending or offline:
        return _offline_result(prompt, answers)
    if history is not None and (not isinstance(history, Sequence) or len(history) > 6
                                or any(not isinstance(item, Mapping) for item in history)):
        raise ValueError("Assistant history is invalid")
    system = (
        "You are a motion direction editor for a humanoid animation model. "
        "Return only JSON matching the schema. Ask up to three short clarifying questions when a specific "
        "movement, direction, reference, or goal remains ambiguous. Otherwise produce one concise refined_prompt "
        "in English, at most 500 characters. Keep every stated constraint (angle, left/right, speed, duration, "
        "body part, in-place movement). Never invent a precise angle, direction, location, or duration. "
        "Explain briefly what changed. Write explanation and questions in the user's language; "
        "option labels should also use that language. If ready is false, refined_prompt must be empty. "
        "If ready is true, questions must be empty. Do not claim that the resulting motion is guaranteed."
    )
    payload = {"original_prompt": prompt, "answers": answers,
               "history": list(history or ()), "language": "Turkish" if _turkish(prompt) else "English"}
    source = "injected" if provider is not None else "ollama"
    try:
        doc = (provider or OllamaPromptProvider(model=model))(system, json.dumps(payload, ensure_ascii=False))
        result = _validate_model_result(doc, source)
        if result.ready:
            _validate_constraints(prompt, answers, result.refined_prompt)
        return result
    except Exception:
        warning = ("Yerel AI kullanılamadı veya geçersiz yanıt verdi; çevrimdışı rehberlik gösteriliyor."
                   if _turkish(prompt) else
                   "Local AI was unavailable or returned an invalid response; showing offline guidance.")
        return _offline_result(prompt, answers, warning)
