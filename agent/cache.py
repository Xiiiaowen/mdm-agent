"""
Simple file-based cache for web search results.
Keyed by normalized query string.
Persists to cache/search_cache.json so repeat runs don't re-call Tavily.
"""

import json
import os

_CACHE_FILE = "cache/search_cache.json"
_cache: dict = {}
_hits = 0
_misses = 0


def _load():
    global _cache
    if os.path.exists(_CACHE_FILE) and not _cache:
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _cache = {}


def _save():
    os.makedirs("cache", exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=2)


def _key(query: str) -> str:
    return query.lower().strip()


def get(query: str):
    global _hits, _misses
    _load()
    result = _cache.get(_key(query))
    if result is not None:
        _hits += 1
    else:
        _misses += 1
    return result


def put(query: str, result: str):
    _load()
    _cache[_key(query)] = result
    _save()


def stats() -> dict:
    return {"size": len(_cache), "hits": _hits, "misses": _misses}
