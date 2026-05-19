"""Clean transcript text for search, LLM prompts, and UI display."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

_ON_SCREEN_RE = re.compile(r"\s*\[On Screen:\s*[^\]]*\]", re.IGNORECASE)
_ON_SCREEN_PAREN_RE = re.compile(r"\s*\(On Screen:\s*[^)]*\)", re.IGNORECASE)
_ON_SCREEN_LOOSE_RE = re.compile(r"\s*On Screen:\s*[^.\[\(]{4,200}", re.IGNORECASE)
_VISUAL_RE = re.compile(r"\s*\[Visual Content\]:\s*[^\[]*", re.IGNORECASE)
_TIME_RANGE_RE = re.compile(r"\s*\[\d+\.?\d*s\s*-\s*\d+\.?\d*s\]\s*$")


def strip_ocr_markup(text: str) -> str:
    """Remove on-screen OCR tags; keep spoken transcript."""
    if not text:
        return ""
    out = text
    for _ in range(8):
        prev = out
        out = _ON_SCREEN_RE.sub("", out)
        out = _ON_SCREEN_PAREN_RE.sub("", out)
        out = _ON_SCREEN_LOOSE_RE.sub("", out)
        out = _VISUAL_RE.sub("", out)
        if out == prev:
            break
    out = collapse_repeated_phrases(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def collapse_repeated_phrases(text: str, min_phrase: int = 40) -> str:
    """Collapse duplicate slide/OCR blobs repeated in one string."""
    text = (text or "").strip()
    if len(text) < min_phrase * 2:
        return text
    # Whole-string repetition (same chunk tiled many times)
    for size in range(min(len(text) // 2, 180), min_phrase - 1, -5):
        chunk = text[:size].strip()
        if len(chunk) < min_phrase:
            continue
        if text.count(chunk) >= 2 and len(chunk) * 2 >= len(text) * 0.45:
            return chunk
    # Sentence-level dedupe (order preserved)
    parts = re.split(r"(?<=[.!?])\s+", text)
    seen: set = set()
    unique: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        key = re.sub(r"\s+", " ", p.lower())[:100]
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    if len(unique) < len(parts):
        return " ".join(unique)
    return text


def clean_query_text(query: str) -> str:
    q = (query or "").strip()
    q = _TIME_RANGE_RE.sub("", q).strip()
    return q or (query or "").strip()


def dedupe_ocr_append(sentence: str, ocr_text: str) -> str:
    """Append OCR once per unique on-screen string (avoids 300× duplicate slide text)."""
    sentence = sentence or ""
    ocr_text = re.sub(r"\s+", " ", (ocr_text or "").strip())
    if not ocr_text or len(ocr_text) < 4:
        return sentence
    if ocr_text.lower() in sentence.lower():
        return sentence
    norm = ocr_text[:80].lower()
    if norm and norm in sentence.lower():
        return sentence
    return f"{sentence} [On Screen: {ocr_text}]".strip()


def merge_ocr_into_sentences(sentences: list, ocr_text_data: list) -> list:
    """Attach OCR to best-matching sentence without duplicating tags."""
    if not ocr_text_data or not sentences:
        return sentences
    for ocr_item in ocr_text_data:
        ts = float(ocr_item.get("timestamp", 0))
        text = ocr_item.get("text") or ""
        best_match = None
        best_dist = float("inf")
        for s in sentences:
            s_start = float(s.get("starttime", 0))
            s_end = float(s.get("endtime", 0))
            if s_start <= ts <= s_end:
                best_match = s
                best_dist = 0
                break
            dist = min(abs(s_start - ts), abs(s_end - ts))
            if dist < 5.0 and dist < best_dist:
                best_dist = dist
                best_match = s
        if best_match is not None:
            best_match["sentence"] = dedupe_ocr_append(
                best_match.get("sentence", ""), text
            )
        else:
            sentences.append(
                {
                    "sentence": f"[Visual Content]: {text}",
                    "starttime": f"{ts:.2f}",
                    "endtime": f"{ts + 2.0:.2f}",
                    "verbs": ["visual_ocr"],
                }
            )
    sentences.sort(key=lambda x: float(x.get("starttime", 0)))
    return sentences


def join_sentences_clean(
    sentences: list,
    start_idx: int,
    end_idx: int,
    max_chars: int = 500,
) -> str:
    parts: List[str] = []
    for t in range(start_idx, end_idx + 1):
        if t < 0 or t >= len(sentences):
            continue
        spoken = strip_ocr_markup(sentences[t].get("sentence", "") or "")
        if spoken and (not parts or spoken != parts[-1]):
            parts.append(spoken)
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def join_sentences_in_range(
    sentences: list,
    start_sec: float,
    end_sec: float,
    max_chars: int = 500,
) -> str:
    parts: List[str] = []
    for s in sentences:
        st = float(s.get("starttime", 0))
        en = float(s.get("endtime", 0))
        if en < start_sec - 1 or st > end_sec + 1:
            continue
        spoken = strip_ocr_markup(s.get("sentence", "") or "")
        if spoken and (not parts or spoken != parts[-1]):
            parts.append(spoken)
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def clip_display_text(
    raw_text: str,
    llm_summary: Optional[str] = None,
    max_chars: int = 400,
) -> str:
    """Prefer Groq summary; fallback to cleaned transcript snippet."""
    summary = (llm_summary or "").strip()
    if summary:
        return summary if len(summary) <= max_chars else summary[: max_chars - 3].rstrip() + "..."
    cleaned = strip_ocr_markup(raw_text or "")
    if not cleaned:
        return "No transcript available for this clip."
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3].rstrip() + "..."


def condensed_transcript_for_llm(sentences: list, max_chars: int = 12000) -> str:
    """Transcript for Groq search — spoken text only, chunked by time."""
    chunks: List[str] = []
    chunk_size = 5
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i : i + chunk_size]
        start_t = group[0].get("starttime", "0")
        end_t = group[-1].get("endtime", "0")
        spoken_parts = [
            strip_ocr_markup(s.get("sentence", "") or "")
            for s in group
        ]
        spoken_parts = [p for p in spoken_parts if p]
        if not spoken_parts:
            continue
        text = " ".join(spoken_parts)
        chunks.append(f"[{start_t}s - {end_t}s]: {text}")
    transcript = "\n".join(chunks)
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n[... transcript truncated ...]"
    return transcript
