"""
Completeness scoring — computed post-processing, no LLM needed.
Measures what % of key business fields are filled in a record.
"""

# Fields that matter for a usable B2B customer record
KEY_FIELDS = [
    "company_name",
    "city",
    "country",
    "industry",
    "contact_email",
    "phone",
    "website",
]


def score(record: dict) -> dict:
    """
    Compute completeness for a single record.

    Returns:
        {
            "score_pct": int (0-100),
            "filled": int,
            "total": int,
            "missing": [field names],
        }
    """
    filled = [f for f in KEY_FIELDS if str(record.get(f, "")).strip()]
    missing = [f for f in KEY_FIELDS if not str(record.get(f, "")).strip()]
    return {
        "score_pct": round(len(filled) / len(KEY_FIELDS) * 100),
        "filled": len(filled),
        "total": len(KEY_FIELDS),
        "missing": missing,
    }


def score_pair(original: dict, corrected: dict) -> dict:
    """Compute before and after completeness scores for a record."""
    before = score(original)
    after = score(corrected)
    return {
        "before": before,
        "after": after,
        "improvement": after["score_pct"] - before["score_pct"],
    }
