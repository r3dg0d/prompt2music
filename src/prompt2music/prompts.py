"""Prompt enhancement, presets and negative prompts.

Enhancement is opt-in (``--enhance`` or ``generation.enhance_prompt``) and
always keeps the user's original words first, appending detail rather than
rewriting, so the requested genre, instruments and mood survive. The
rule-based path needs no model; a local LLM (Ollama / llama.cpp /
transformers) can be configured in ``[llm]`` and is validated the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

GENRES: dict[str, dict[str, Any]] = {
    "ambient": {"desc": "slowly evolving textures, wide stereo pads, soft attacks",
                "bpm": (60, 90), "inst": ["warm pads", "granular textures"],
                "neg": ["harsh transients", "busy drums"]},
    "synthwave": {"desc": "retro analog synth arpeggios, gated reverb drums, 80s production",
                  "bpm": (80, 118), "inst": ["analog synth bass", "lush polysynth pads"],
                  "neg": ["acoustic guitar", "lo-fi hiss"]},
    "cyberpunk": {"desc": "dark futuristic sound design, distorted bass, glitchy details",
                  "bpm": (95, 130), "inst": ["distorted synth bass", "metallic percussion"],
                  "neg": ["acoustic folk instruments"]},
    "techno": {"desc": "driving four-on-the-floor kick, hypnotic repetition, warehouse mix",
               "bpm": (125, 150), "inst": ["punchy kick drum", "rolling bassline", "hi-hats"],
               "neg": ["vocals", "acoustic guitar"]},
    "industrial": {"desc": "abrasive distorted textures, mechanical rhythms, heavy compression",
                   "bpm": (110, 150), "inst": ["distorted drums", "noise layers"],
                   "neg": ["soft acoustic textures"]},
    "house": {"desc": "groovy four-on-the-floor beat, warm bass, uplifting chords",
              "bpm": (118, 128), "inst": ["piano stabs", "deep bass"], "neg": ["harsh noise"]},
    "drum and bass": {"desc": "fast breakbeats, deep sub bass, energetic rolling groove",
                      "bpm": (165, 178), "inst": ["breakbeat drums", "reese bass"],
                      "neg": ["slow tempo"]},
    "lofi": {"desc": "dusty vinyl texture, mellow jazzy chords, laid-back swing",
             "bpm": (70, 90), "inst": ["rhodes piano", "boom-bap drums"],
             "neg": ["aggressive distortion"]},
    "hip hop": {"desc": "punchy boom-bap or trap drums, prominent bass, rhythmic groove",
                "bpm": (80, 100), "inst": ["808 bass", "crisp snare"], "neg": ["muddy mix"]},
    "orchestral": {"desc": "full symphonic arrangement, dynamic swells, concert hall reverb",
                   "bpm": (70, 140), "inst": ["strings", "brass", "timpani"],
                   "neg": ["synthesizers", "electronic drums"]},
    "cinematic": {"desc": "epic film-score build, layered strings and percussion, wide dynamics",
                  "bpm": (70, 130), "inst": ["string ostinatos", "taiko drums", "brass"],
                  "neg": ["lo-fi", "thin mix"]},
    "piano": {"desc": "expressive solo piano, natural dynamics, intimate room sound",
              "bpm": (60, 100), "inst": ["grand piano"], "neg": ["drums", "distortion"]},
    "jazz": {"desc": "swinging rhythm section, extended chords, improvised solos",
             "bpm": (90, 180), "inst": ["upright bass", "brushed drums", "piano"],
             "neg": ["electronic drums"]},
    "rock": {"desc": "driving electric guitars, live drum kit, energetic band performance",
             "bpm": (100, 150), "inst": ["electric guitar", "bass guitar", "drum kit"],
             "neg": ["synth pads"]},
    "metal": {"desc": "heavily distorted guitars, double-kick drums, aggressive riffs",
              "bpm": (120, 200), "inst": ["distorted guitars", "double bass drum"],
              "neg": ["soft ballad"]},
    "folk": {"desc": "acoustic instrumentation, organic warm recording, storytelling feel",
             "bpm": (80, 120), "inst": ["acoustic guitar", "fiddle"],
             "neg": ["electronic beats"]},
    "chiptune": {"desc": "8-bit square and triangle waves, retro game console sound",
                 "bpm": (120, 160), "inst": ["square wave lead", "noise channel drums"],
                 "neg": ["orchestral reverb"]},
    "funk": {"desc": "tight syncopated groove, slap bass, horn stabs",
             "bpm": (95, 115), "inst": ["slap bass", "clavinet", "horn section"],
             "neg": ["ambient drones"]},
}
GENRE_ALIASES = {"dnb": "drum and bass", "lo-fi": "lofi", "lo fi": "lofi", "edm": "house",
                 "classical": "orchestral", "film score": "cinematic", "soundtrack": "cinematic",
                 "trap": "hip hop", "boss battle": "cinematic", "8-bit": "chiptune"}

MOODS: dict[str, str] = {
    "sad": "melancholic, introspective",
    "melancholic": "wistful, bittersweet",
    "happy": "uplifting, bright and joyful",
    "dark": "brooding, ominous",
    "epic": "grand, heroic and triumphant",
    "dreamy": "ethereal, floating",
    "calm": "peaceful, gentle",
    "aggressive": "intense, relentless",
    "romantic": "tender, warm",
    "mysterious": "enigmatic, suspenseful",
    "energetic": "high-energy, propulsive",
    "nostalgic": "warm, nostalgic",
    "tense": "suspenseful, uneasy",
    "chill": "relaxed, laid-back",
}
INSTRUMENTS = ("piano", "guitar", "strings", "violin", "cello", "synth", "synthesizer", "bass",
               "drums", "percussion", "brass", "trumpet", "saxophone", "flute", "organ", "choir",
               "harp", "pads", "808", "rhodes", "kick", "vocals", "marimba", "bells")
BASE_NEGATIVE = ["low quality", "muffled", "digital clipping", "off-key", "noise artifacts"]
KEY_RE = re.compile(r"\b([A-G](?:#|b)?)(m|min|minor|maj|major)?\b")


@dataclass
class Enhanced:
    original: str
    prompt: str
    negative_prompt: str
    detected: dict[str, Any] = field(default_factory=dict)
    method: str = "rules"

    def as_dict(self) -> dict[str, Any]:
        return {"original": self.original, "prompt": self.prompt,
                "negative_prompt": self.negative_prompt, "detected": self.detected,
                "method": self.method}


PRESETS: dict[str, dict[str, Any]] = {
    "boss-battle": {"prompt": "Epic orchestral boss battle music with aggressive strings, "
                    "pounding cinematic percussion and heroic brass", "bpm": 150},
    "synthwave-night": {"prompt": "Dark synthwave night drive with pulsing analog bass, "
                        "gated snare and shimmering arpeggios", "bpm": 100},
    "lofi-study": {"prompt": "Mellow lofi hip hop beat with dusty vinyl crackle, jazzy rhodes "
                   "chords and laid-back drums", "bpm": 80},
    "ambient-drift": {"prompt": "Dreamy ambient electronic music with shimmering synthesizers "
                      "and slow evolving textures", "bpm": 70},
    "warehouse-techno": {"prompt": "Aggressive industrial techno with distorted kick drums, "
                         "metallic percussion and a dark atmosphere", "bpm": 145},
    "piano-intimate": {"prompt": "Intimate solo piano with delicate arpeggios and gentle room "
                       "reverb", "bpm": 72},
    "chiptune-level": {"prompt": "Upbeat chiptune level theme with catchy square-wave melody "
                       "and driving noise drums", "bpm": 140},
    "cinematic-trailer": {"prompt": "Cinematic trailer build with rising string ostinatos, "
                          "taiko drums and a massive brass climax", "bpm": 120},
}


def detect(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    genres = []
    for alias, target in GENRE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text) and target not in genres:
            genres.append(target)
    for name in GENRES:
        if re.search(rf"\b{re.escape(name)}\b", text) and name not in genres:
            genres.append(name)
    moods = [m for m in MOODS if re.search(rf"\b{m}\b", text)]
    instruments = [i for i in INSTRUMENTS if re.search(rf"\b{re.escape(i)}s?\b", text)]
    bpm = None
    m = re.search(r"\b(\d{2,3})\s*bpm\b", text)
    if m:
        bpm = int(m.group(1))
    key = None
    km = re.search(r"\bin ([A-G](?:#|b)?)\s*(minor|major|m)?\b", prompt)
    if km:
        key = km.group(1) + (" minor" if km.group(2) in ("minor", "m") else
                             " major" if km.group(2) else "")
    return {"genres": genres, "moods": moods, "instruments": instruments, "bpm": bpm,
            "key": key}


def _stem(word: str) -> str:
    """Crude prefix stem so 'distorted' and 'distortion' compare equal."""
    return word[:6]


def negative_for(detected: dict[str, Any], *, instrumental: bool,
                 prompt: str = "") -> str:
    """Genre-aware negatives that never contradict something the user asked for."""
    neg = list(BASE_NEGATIVE)
    asked = {_stem(w) for w in re.findall(r"[a-z0-9]+", prompt.lower())}
    asked |= {_stem(w) for i in detected.get("instruments", []) for w in i.split()}
    for genre in detected.get("genres", []):
        for item in GENRES[genre]["neg"]:
            words = {_stem(w) for w in re.findall(r"[a-z0-9]+", item)}
            if item not in neg and not words & asked:
                neg.append(item)
    if not instrumental:
        neg = [n for n in neg if n != "vocals"]
    elif "vocals" not in neg and _stem("vocals") not in asked:
        neg.append("vocals")
    return ", ".join(neg)


def enhance_rules(prompt: str, *, instrumental: bool = True, bpm: int | None = None,
                  key: str | None = None) -> Enhanced:
    original = " ".join(prompt.split())
    found = detect(original)
    extras: list[str] = []
    for mood in found["moods"]:
        extras.append(MOODS[mood])
    for genre in found["genres"][:2]:
        extras.append(GENRES[genre]["desc"])
    have = " ".join(found["instruments"])
    for genre in found["genres"][:1]:
        add = [i for i in GENRES[genre]["inst"] if i.split()[-1] not in have]
        if add and not found["instruments"]:
            extras.append("featuring " + ", ".join(add))
    if not found["genres"] and not found["moods"]:
        extras.append("rich, well-balanced arrangement, clear stereo mix")
    tempo = bpm or found["bpm"]
    if tempo and not found["bpm"]:
        extras.append(f"{tempo} BPM")
    if key and not found["key"]:
        extras.append(f"in {key}")
    extras.append("instrumental" if instrumental else "with vocals")
    extras.append("high-quality studio production")
    seen: set[str] = set()
    parts = []
    for part in extras:
        if part.lower() not in seen and part.lower() not in original.lower():
            parts.append(part)
            seen.add(part.lower())
    enhanced = original.rstrip(" .,") + (", " + ", ".join(parts) if parts else "")
    return Enhanced(original, enhanced,
                    negative_for(found, instrumental=instrumental, prompt=original), found)


def preserves_intent(original: str, candidate: str) -> bool:
    """An enhancement must keep every detected genre, mood and instrument."""
    a, b = detect(original), detect(candidate)
    return all(set(a[k]) <= set(b[k]) for k in ("genres", "moods", "instruments"))


def enhance(prompt: str, *, instrumental: bool = True, bpm: int | None = None,
            key: str | None = None, llm: Any = None) -> Enhanced:
    """Rule-based by default; an LLM result is used only if it preserves intent."""
    base = enhance_rules(prompt, instrumental=instrumental, bpm=bpm, key=key)
    if llm is None:
        return base
    from prompt2music.llm import ENHANCE_SYSTEM

    text = llm.complete(ENHANCE_SYSTEM, f"Description: {prompt}\nInstrumental: {instrumental}")
    text = " ".join(text.strip().strip('"').split())
    if text and len(text) <= 500 and preserves_intent(prompt, text):
        return Enhanced(base.original, text, base.negative_prompt, base.detected, "llm")
    base.method = "rules (llm output rejected: changed genre/mood/instruments or too long)"
    return base


def apply_preset(name: str, subject: str | None = None) -> dict[str, Any]:
    preset = PRESETS.get(name)
    if preset is None:
        from prompt2music.errors import InvalidRequestError

        raise InvalidRequestError(f"unknown preset {name!r}; available: {', '.join(PRESETS)}")
    out = dict(preset)
    if subject:
        out["prompt"] = f"{out['prompt']}, {subject}"
    return out


def fold_controls_into_prompt(prompt: str, *, bpm: int | None, key: str | None,
                              time_signature: str | None = None) -> str:
    """For models without dedicated BPM/key inputs, state them in the text."""
    found = detect(prompt)
    extra = []
    if bpm and not found["bpm"]:
        extra.append(f"{bpm} BPM")
    if key and not found["key"] and key.lower() not in prompt.lower():
        extra.append(f"in {key}")
    if time_signature and time_signature not in prompt:
        extra.append(f"{time_signature} time")
    return prompt.rstrip(" .,") + (", " + ", ".join(extra) if extra else "")
