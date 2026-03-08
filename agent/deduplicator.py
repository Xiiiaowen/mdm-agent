"""
Deduplication pre-pass — runs BEFORE the agent enrichment loop.

Phase 0: Normalize all company names to English (batch GPT call for non-ASCII names)
Phase 1: rapidfuzz finds candidate pairs on the English names (cheap, no API calls)
Phase 2: GPT-4o-mini verifies each candidate pair (one small LLM call per pair)

Returns which records to discard so the agent never enriches a duplicate.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
# token_set_ratio handles "3M" vs "3M Company" or "BASF" vs "BASF SE"
# token_sort_ratio would score these ~33% (fails) because one name is much shorter

load_dotenv(override=True)
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Junk detection ────────────────────────────────────────────────────────────

_JUNK_NAMES = {"test", "n/a", "na", "none", "unknown", "placeholder", "dummy",
               "fake", "delete", "xxx", "sample", "temp", "tbd", "-", "--"}

_JUNK_PHRASES = ["test entry", "do not use", "unknown company", "placeholder company",
                 "some street", "somewhere"]


def detect_junk(records: list) -> set:
    """
    Fast heuristic junk filter — no API call needed.
    Returns a set of indices for records that are clearly test/placeholder data.

    Rules (name-based only):
    - Company name exactly matches a known junk keyword (e.g. "test", "n/a", "unknown")
    - Company name contains a junk phrase (e.g. "test entry", "do not use")

    We deliberately do NOT flag records just because they have many empty fields —
    that is low completeness, not junk. The agent's job is to enrich those records.
    """
    junk_indices = set()
    for i, record in enumerate(records):
        name = str(record.get("company_name", "")).lower().strip()

        if name in _JUNK_NAMES:
            junk_indices.add(i)
            continue

        if any(phrase in name for phrase in _JUNK_PHRASES):
            junk_indices.add(i)

    return junk_indices


# ── Name normalisation ────────────────────────────────────────────────────────

def _normalize_names_to_english(records: list) -> list:
    """
    Return company names normalized to lowercase English for fuzzy matching.
    Non-ASCII names (Chinese, Japanese, etc.) are translated in one batch GPT call.
    Original records are not modified.
    """
    names = [str(r.get("company_name", "")).strip() for r in records]

    # Find indices with non-ASCII names (Chinese, Korean, Japanese, Arabic, etc.)
    to_translate = {i: name for i, name in enumerate(names) if name and not name.isascii()}

    if to_translate:
        prompt = (
            "Translate these company names to their official English names. "
            "Return JSON only, keys are the numbers, values are the English names.\n\n"
            + "\n".join(f"{i}: {name}" for i, name in to_translate.items())
        )
        try:
            response = _client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            match = re.search(r"\{[\s\S]*\}", text)
            translations = json.loads(match.group()) if match else {}
        except (json.JSONDecodeError, AttributeError, Exception):
            translations = {}

        for i, original in to_translate.items():
            names[i] = translations.get(str(i), original)

    return [n.lower().strip() for n in names]


def _find_candidate_pairs(normalized_names: list, threshold: int = 75) -> list:
    """O(n²) fuzzy scan on English-normalized names."""
    pairs = []
    seen = set()
    for i in range(len(normalized_names)):
        for j in range(i + 1, len(normalized_names)):
            if (i, j) in seen or not normalized_names[i] or not normalized_names[j]:
                continue
            score = fuzz.token_set_ratio(normalized_names[i], normalized_names[j])
            if score >= threshold:
                pairs.append((i, j, score))
                seen.add((i, j))
    return sorted(pairs, key=lambda x: -x[2])


def _verify_pair(rec_a: dict, rec_b: dict) -> dict:
    """Quick LLM call to decide if two records are the same company."""
    prompt = (
        "Are these two B2B customer records the same real-world company?\n\n"
        f"Record A: {json.dumps(rec_a, ensure_ascii=False)}\n\n"
        f"Record B: {json.dumps(rec_b, ensure_ascii=False)}\n\n"
        'Reply with JSON only:\n'
        '{"same_company": true/false, "keep": "A" or "B" or "both", "reason": "one sentence"}\n\n'
        '"keep" = the more complete / canonical record to keep. '
        'If not the same company, use "both".'
    )
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        text = response.choices[0].message.content or ""
        match = re.search(r"\{[\s\S]*\}", text)
        return json.loads(match.group()) if match else {"same_company": False, "keep": "both"}
    except (json.JSONDecodeError, AttributeError):
        return {"same_company": False, "keep": "both"}


def run(records: list, threshold: int = 75) -> dict:
    """
    Full deduplication pre-pass.

    Returns:
        {
            "pairs": [
                {
                    "idx_a", "idx_b", "name_a", "name_b",
                    "similarity", "same_company", "keep", "reason"
                }, ...
            ],
            "discard_indices": set of record indices to skip in the agent loop,
        }
    """
    # Phase 0: normalize all names to English for accurate fuzzy matching
    normalized_names = _normalize_names_to_english(records)

    # Phase 1: fast fuzzy scan
    candidate_pairs = _find_candidate_pairs(normalized_names, threshold)

    # Phase 2: LLM verification
    verified_pairs = []
    discard_indices: set[int] = set()

    for idx_a, idx_b, score in candidate_pairs:
        if idx_a in discard_indices or idx_b in discard_indices:
            continue

        verdict = _verify_pair(records[idx_a], records[idx_b])
        pair_info = {
            "idx_a": idx_a,
            "idx_b": idx_b,
            "name_a": records[idx_a].get("company_name", ""),
            "name_b": records[idx_b].get("company_name", ""),
            "similarity": score,
            "same_company": verdict.get("same_company", False),
            "keep": verdict.get("keep", "both"),
            "reason": verdict.get("reason", ""),
        }
        verified_pairs.append(pair_info)

        if verdict.get("same_company"):
            if verdict.get("keep") == "A":
                discard_indices.add(idx_b)
            elif verdict.get("keep") == "B":
                discard_indices.add(idx_a)
            # "both" = uncertain, keep both and let the agent handle it

    return {"pairs": verified_pairs, "discard_indices": discard_indices}
