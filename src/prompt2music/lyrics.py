"""Structured lyrics: templates, validation and optional LLM drafting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from prompt2music.errors import InvalidRequestError

SECTION_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
KNOWN_SECTIONS = ("intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "hook",
                  "breakdown", "drop", "interlude", "solo", "outro", "instrumental", "inst")
STRUCTURES = {
    "pop": ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus",
            "Bridge", "Chorus", "Outro"],
    "standard": ["Intro", "Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus",
                 "Outro"],
    "short": ["Verse 1", "Chorus", "Verse 2", "Chorus"],
    "ballad": ["Intro", "Verse 1", "Verse 2", "Chorus", "Verse 3", "Chorus", "Outro"],
}
MODEL_LIMITS = {"ace-step": 4096, "heartmula": 8000, "yue": 8000}


@dataclass
class LyricsReport:
    ok: bool
    sections: list[str]
    lines: int
    characters: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def template(structure: str = "standard", *, topic: str | None = None) -> str:
    sections = STRUCTURES.get(structure)
    if sections is None:
        raise InvalidRequestError(f"unknown structure {structure!r}; "
                                  f"choose from {', '.join(STRUCTURES)}")
    out = []
    if topic:
        out.append(f"# Topic: {topic}")
    for name in sections:
        out.append(f"[{name}]")
        if name.lower() in ("intro", "outro", "interlude", "instrumental"):
            out.append("")
        else:
            out.extend(["(write line 1)", "(write line 2)", "(write line 3)",
                        "(write line 4)", ""])
    return "\n".join(out).rstrip() + "\n"


def strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def validate(text: str, *, backend: str | None = None) -> LyricsReport:
    body = strip_comments(text).strip()
    sections: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    lines = 0
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            name = m.group("name").strip()
            sections.append(name)
            base = re.sub(r"\s*\d+$", "", name.lower())
            if base not in KNOWN_SECTIONS:
                warnings.append(f"unusual section tag [{name}]")
            continue
        if "[" in line or "]" in line:
            warnings.append(f"stray bracket in lyric line: {line[:40]!r}")
        if "(write line" in line:
            errors.append("template placeholders are still present")
        lines += 1
    if not body:
        errors.append("lyrics are empty")
    if body and not sections:
        warnings.append("no section tags; add [Verse 1], [Chorus], ... for better structure")
    limit = MODEL_LIMITS.get(backend or "")
    if limit and len(body) > limit:
        errors.append(f"{backend} accepts at most {limit} characters (got {len(body)})")
    return LyricsReport(not errors, sections, lines, len(body), sorted(set(warnings)),
                        sorted(set(errors)))


def generate_lyrics(llm: Any, prompt: str, *, structure: str = "standard",
                    genre: str | None = None, mood: str | None = None,
                    language: str = "English") -> str:
    sections = ", ".join(STRUCTURES.get(structure, STRUCTURES["standard"]))
    user = (f"Topic: {prompt}\nGenre: {genre or 'any'}\nMood: {mood or 'any'}\n"
            f"Language: {language}\nSections in order: {sections}")
    from prompt2music.llm import LYRICS_SYSTEM

    text = llm.complete(LYRICS_SYSTEM, user, max_tokens=900).strip()
    report = validate(text)
    if not report.sections:
        raise InvalidRequestError("the language model did not return tagged lyrics; try again "
                                  "or start from `prompt2music lyrics template`")
    return text + "\n"
