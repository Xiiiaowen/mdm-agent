"""
Tool implementations available to the MDM agent.
Each function maps to a tool the agent can choose to call.
"""

import json
import os
import re

import phonenumbers
import requests
from tavily import TavilyClient

from . import cache as search_cache

_tavily = None


def _get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily

_FAKE_DOMAINS = {"test.com", "example.com", "fake.com", "noreply.com", "placeholder.com"}


def web_search(query: str) -> str:
    """Search the web for company information. Checks cache first."""
    cached = search_cache.get(query)
    if cached:
        return cached
    results = _get_tavily().search(query=query, max_results=3)
    snippets = [r["content"] for r in results.get("results", []) if r.get("content")]
    result = "\n\n---\n\n".join(snippets) if snippets else "No results found."
    search_cache.put(query, result)
    return result


def validate_phone(phone: str, country_code: str = "") -> dict:
    """Validate and normalize a phone number to E.164 format."""
    if not phone:
        return {"valid": False, "reason": "No phone number provided."}
    try:
        parsed = phonenumbers.parse(phone, country_code.upper() if country_code else None)
        if phonenumbers.is_valid_number(parsed):
            return {
                "valid": True,
                "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                "international": phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                ),
            }
        return {"valid": False, "reason": "Number is not valid for the given region."}
    except phonenumbers.NumberParseException as e:
        return {"valid": False, "reason": str(e)}


def validate_email(email: str) -> dict:
    """Check whether an email address is properly formatted and not a test address."""
    email = email.strip()
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        return {"valid": False, "reason": "Invalid email format."}
    domain = email.split("@")[1].lower()
    if domain in _FAKE_DOMAINS:
        return {"valid": False, "reason": f"Looks like a test/placeholder address (domain: {domain})."}
    return {"valid": True, "normalized": email.lower()}


def gleif_lookup(company_name: str, country_code: str = "") -> dict:
    """
    Look up a company in the GLEIF LEI registry (free, no API key).
    Returns up to 5 candidates (ASCII names only) so the agent can pick the best match.
    """
    params = {
        "filter[entity.legalName]": company_name,
        "page[number]": 1,
        "page[size]": 10,
    }
    if country_code:
        params["filter[entity.legalAddress.country]"] = country_code.upper()

    cache_key = f"gleif:{company_name.lower()}:{country_code.lower()}"
    cached = search_cache.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        resp = requests.get(
            "https://api.gleif.org/api/v1/lei-records",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        return {"found": False, "reason": str(e)}

    if not data:
        return {"found": False, "reason": "No matching entity found in GLEIF registry."}

    candidates = []
    for item in data:
        attrs = item["attributes"]
        entity = attrs["entity"]
        name = entity["legalName"]["name"]

        # Skip non-ASCII names (Chinese, Japanese, etc.) — keep English names only
        if not name.isascii():
            continue

        hq = entity.get("headquartersAddress") or entity.get("legalAddress") or {}
        candidates.append({
            "lei": attrs["lei"],
            "legal_name": name,
            "status": entity.get("status", ""),
            "jurisdiction": entity.get("jurisdiction", ""),
            "address": ", ".join(filter(None, hq.get("addressLines", []))),
            "city": hq.get("city", ""),
            "country": hq.get("country", ""),
            "postal_code": hq.get("postalCode", ""),
            "registration_id": entity.get("registeredAs", ""),
        })
        if len(candidates) == 5:
            break

    if not candidates:
        return {"found": False, "reason": "No ASCII-name entity found in GLEIF registry."}

    result = {
        "found": True,
        "note": "Multiple candidates returned — pick the best match for the company.",
        "candidates": candidates,
    }
    search_cache.put(cache_key, json.dumps(result, ensure_ascii=False))
    return result


def flag_for_review(field: str, reason: str) -> dict:
    """Flag a field as unresolvable — needs human review."""
    return {"flagged": True, "field": field, "reason": reason}


# Maps tool name → callable, used by the agent loop
TOOL_REGISTRY = {
    "web_search": lambda inp: web_search(inp["query"]),
    "gleif_lookup": lambda inp: gleif_lookup(inp["company_name"], inp.get("country_code", "")),
    "validate_phone": lambda inp: validate_phone(inp.get("phone") or inp.get("number") or inp.get("phone_number", ""), inp.get("country_code", "")),
    "validate_email": lambda inp: validate_email(inp["email"]),
    "flag_for_review": lambda inp: flag_for_review(inp["field"], inp["reason"]),
}
