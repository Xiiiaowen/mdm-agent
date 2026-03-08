"""
MDM Agent — the real agent loop.

GPT-4o-mini autonomously decides which tools to call for each record,
observes the results, and loops until it has a final answer.

Note: deduplication is handled as a pre-pass (see deduplicator.py) before
this runs, so the agent only processes surviving, non-duplicate records.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from .tools import TOOL_REGISTRY

load_dotenv(override=True)
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web to find or verify company information: "
                "official name, headquarters address, phone, email, website, industry, country. "
                "Use specific queries like 'Siemens AG Frankfurt headquarters phone number'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A precise search query targeting what you need to verify or fill in.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_phone",
            "description": "Validate and normalize a phone number to E.164 format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "Raw phone number string."},
                    "country_code": {
                        "type": "string",
                        "description": "ISO 3166-1 alpha-2 country code (e.g. 'DE', 'CN', 'US'). Include if known.",
                    },
                },
                "required": ["phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_email",
            "description": (
                "Check whether an email address is correctly formatted and not a test/placeholder address. "
                "Use this whenever an email is present in the record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email address to validate."},
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_review",
            "description": (
                "Flag a specific field as unresolvable after exhausting your investigation. "
                "Use this when you cannot confidently determine the correct value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "The field name that cannot be resolved."},
                    "reason": {"type": "string", "description": "Why the field cannot be resolved."},
                },
                "required": ["field", "reason"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a B2B customer data quality investigator. You receive a customer record that may be incomplete, incorrect, or contain messy data. Duplicates have already been removed upstream — focus on cleaning and enriching this record.

Your job:
1. Analyse the record — identify what is missing, inconsistent, or suspicious.
2. Decide which fields need investigation (only investigate what actually needs it).
3. Use your tools to verify or fill in missing information. You may call tools multiple times.
4. If a field truly cannot be resolved, flag it for human review.
5. Produce a corrected and enriched final record.

What to check:
- Company name: standardise to official English name if in another language or abbreviated
- Country: must be ISO 3166-1 alpha-2 (e.g. "Germany" → "DE", "China" → "CN")
- Phone: validate and normalise to E.164 format if present
- Email: validate format and detect test/fake addresses
- Missing fields: use web_search to fill in address, city, country, industry, phone, website
- Junk records: if clearly test data (e.g. "TEST ENTRY", all fields "N/A"), flag the entire record

At the end, output ONLY a valid JSON object with this exact structure:
{
  "corrected_record": { ...all original fields with corrections applied... },
  "changes": ["list of strings describing each change made and why"],
  "flags": ["field: reason" for each unresolved field],
  "is_junk": true or false,
  "confidence": "high" or "medium" or "low"
}"""


def investigate(record: dict) -> dict:
    """
    Run the agent on a single record.

    Returns:
        {
            "result": { corrected_record, changes, flags, is_junk, confidence },
            "trace": [ { type, ... } ]
        }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Investigate and fix this customer record:\n\n"
                + json.dumps(record, ensure_ascii=False, indent=2)
            ),
        },
    ]
    trace = []

    while True:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if message.content and message.content.strip():
            trace.append({"type": "thought", "content": message.content.strip()})

        if finish_reason == "stop":
            result = _parse_json_result(message.content or "", record)
            return {"result": result, "trace": trace}

        if finish_reason == "tool_calls":
            messages.append(message)
            for tc in message.tool_calls:
                tool_input = json.loads(tc.function.arguments)
                output = TOOL_REGISTRY[tc.function.name](tool_input)
                trace.append(
                    {
                        "type": "tool_call",
                        "tool": tc.function.name,
                        "input": tool_input,
                        "output": output,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            json.dumps(output, ensure_ascii=False)
                            if not isinstance(output, str)
                            else output
                        ),
                    }
                )


def _parse_json_result(text: str, original_record: dict) -> dict:
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {
        "corrected_record": original_record,
        "changes": [],
        "flags": ["parse_error: Agent response could not be parsed"],
        "is_junk": False,
        "confidence": "low",
    }
