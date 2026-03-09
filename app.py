"""
MDM Agent — Streamlit UI

Flow:
  1. Upload file
  2. Pre-dedup pass (fast fuzzy + LLM verification) — removes duplicates before enrichment
  3. Agent investigates each surviving record (web search cached)
  4. Results with completeness scores, duplicate summary, download
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from agent.investigator import investigate
from agent.completeness import score_pair
from agent import deduplicator
from agent import cache as search_cache

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MDM Agent", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.trace-thought { background:#f0f4ff; border-left:3px solid #4a6cf7; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:0.88rem; }
.trace-tool    { background:#fff8e1; border-left:3px solid #f5a623; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:0.88rem; }
.badge-high    { background:#d4edda; color:#155724; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-medium  { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-low     { background:#f8d7da; color:#721c24; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-junk    { background:#6c757d; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 MDM Agent")
st.caption(
    "A real AI agent for B2B customer data quality. "
    "Deduplicates first, then GPT-4o-mini investigates each surviving record — "
    "calling web search, phone/email validation, or human escalation as needed."
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("results", []),
    ("dedup_result", None),
    ("original_df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "company_name", "address", "city", "country", "industry",
    "contact_email", "phone", "website", "registration_id",
]


def _load_file(uploaded) -> pd.DataFrame:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded, dtype=str).fillna("")
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(uploaded, dtype=str).fillna("")
    elif ext == "json":
        return pd.read_json(uploaded, dtype=str).fillna("")
    raise ValueError(f"Unsupported file type: {ext}")


def _render_trace(trace: list):
    for step in trace:
        if step["type"] == "thought":
            st.markdown(
                f'<div class="trace-thought">💭 {step["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif step["type"] == "tool_call":
            tool = step["tool"]
            inp = json.dumps(step["input"], ensure_ascii=False)
            out = step["output"]
            out_str = json.dumps(out, ensure_ascii=False) if not isinstance(out, str) else out
            if len(out_str) > 500:
                out_str = out_str[:500] + "…"
            st.markdown(
                f'<div class="trace-tool">'
                f'🔧 <b>{tool}</b>({inp})<br>'
                f'<span style="color:#555">→ {out_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_diff(original: dict, corrected: dict):
    rows = []
    for field in EXPECTED_COLUMNS:
        orig = str(original.get(field, "")).strip()
        corr = str(corrected.get(field, "")).strip()
        rows.append({
            "Field": field,
            "Before": orig,
            "After": corr,
            "Changed": "✓" if orig != corr else "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.apply(
            lambda row: ["background-color:#e8f5e9" if row["Changed"] == "✓" else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )


def _confidence_badge(conf: str) -> str:
    cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(conf, "badge-low")
    return f'<span class="{cls}">{conf.upper()}</span>'


def _completeness_bar(pct: int) -> str:
    color = "#28a745" if pct >= 70 else "#ffc107" if pct >= 40 else "#dc3545"
    return (
        f'<div style="background:#eee;border-radius:6px;height:10px;width:100%;">'
        f'<div style="background:{color};width:{pct}%;height:10px;border-radius:6px;"></div>'
        f'</div><small>{pct}%</small>'
    )


def _build_output_df(results: list) -> pd.DataFrame:
    rows = []
    for item in results:
        rec = dict(item["result"].get("corrected_record", item["original"]))
        rec["_changes"] = " | ".join(item["result"].get("changes", []))
        rec["_flags"] = " | ".join(item["result"].get("flags", []))
        rec["_confidence"] = item["result"].get("confidence", "")
        rec["_is_junk"] = item["result"].get("is_junk", False)
        rec["_completeness_before"] = item["completeness"]["before"]["score_pct"]
        rec["_completeness_after"] = item["completeness"]["after"]["score_pct"]
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown("**Phase 1** — Junk filter _(no API cost)_")
    st.markdown("**Phase 2** — Dedup: fuzzy + LLM verification")
    st.markdown("**Phase 3** — Agent enriches survivors only")
    st.divider()
    st.markdown("**Agent tools**")
    st.markdown("🏛️ `gleif_lookup` — GLEIF LEI registry")
    st.markdown("🌐 `web_search` — Tavily (cached)")
    st.markdown("📞 `validate_phone` — E.164 format")
    st.markdown("📧 `validate_email` — format + fake check")
    st.markdown("🚩 `flag_for_review` — human escalation")
    st.divider()
    cache_stats = search_cache.stats()
    st.markdown("**Search cache**")
    st.caption(f"{cache_stats['size']} entries · {cache_stats['hits']} hits · {cache_stats['misses']} misses")
    st.caption("Powered by GPT-4o-mini + Tavily + GLEIF")


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload customer records",
    type=["csv", "xlsx", "xls", "json"],
    help="Expected columns: company_name, address, city, country, industry, contact_email, phone, website, registration_id",
)

if uploaded:
    try:
        df = _load_file(uploaded)
        st.session_state.original_df = df
        st.session_state.results = []
        st.session_state.dedup_result = None
        st.success(f"Loaded {len(df)} records from `{uploaded.name}`")
        with st.expander("Preview raw data", expanded=False):
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load file: {e}")

# ── Run ───────────────────────────────────────────────────────────────────────
if st.session_state.original_df is not None:
    df = st.session_state.original_df

    col1, col2 = st.columns([2, 1])
    with col1:
        max_records = st.slider(
            "Records to process", min_value=1, max_value=len(df), value=min(10, len(df))
        )
    with col2:
        run = st.button("▶ Run Agent", type="primary", use_container_width=True)

    if run:
        st.session_state.results = []
        all_records = df.head(max_records).to_dict(orient="records")

        # ── Phase 1: Junk filter (heuristic, no API call) ────────────────────
        st.markdown("### Phase 1 — Junk Filter")
        junk_indices = deduplicator.detect_junk(all_records)
        clean_records = [(i, r) for i, r in enumerate(all_records) if i not in junk_indices]

        j1, j2 = st.columns(2)
        j1.metric("Junk / test records removed", len(junk_indices))
        j2.metric("Records remaining", len(clean_records))

        if junk_indices:
            with st.expander(f"Junk records removed ({len(junk_indices)})", expanded=True):
                for i in sorted(junk_indices):
                    name = all_records[i].get("company_name", f"Record {i+1}")
                    st.markdown(f"- **{name}** — detected as test/placeholder data")

        # ── Phase 2: Deduplication pre-pass ──────────────────────────────────
        st.markdown("### Phase 2 — Deduplication")
        clean_only = [r for _, r in clean_records]
        with st.spinner(f"Scanning {len(clean_only)} records for duplicates…"):
            dedup = deduplicator.run(clean_only)
            st.session_state.dedup_result = dedup

        discard = dedup["discard_indices"]
        pairs = dedup["pairs"]
        same = [p for p in pairs if p["same_company"]]
        diff = [p for p in pairs if not p["same_company"]]
        survivors = [r for i, r in enumerate(clean_only) if i not in discard]

        d1, d2, d3 = st.columns(3)
        d1.metric("Candidate pairs found", len(pairs))
        d2.metric("Confirmed duplicates", len(same))
        d3.metric("Surviving records", len(survivors))

        if same:
            with st.expander(f"Duplicates removed ({len(same)})", expanded=True):
                for p in same:
                    kept = p["name_a"] if p["keep"] == "A" else p["name_b"]
                    discarded = p["name_b"] if p["keep"] == "A" else p["name_a"]
                    st.markdown(
                        f"- **{discarded}** → duplicate of **{kept}** "
                        f"(similarity {p['similarity']}%) · _{p['reason']}_"
                    )

        if diff:
            with st.expander(f"Similar but different companies ({len(diff)})", expanded=False):
                for p in diff:
                    st.markdown(
                        f"- **{p['name_a']}** vs **{p['name_b']}** "
                        f"(similarity {p['similarity']}%) → kept both · _{p['reason']}_"
                    )

        # ── Phase 3: Agent investigation ──────────────────────────────────────
        st.markdown("### Phase 3 — Agent Investigation")
        progress = st.progress(0, text="Starting agent…")
        status = st.empty()

        for i, record in enumerate(survivors):
            company = record.get("company_name", f"Record {i+1}")
            status.markdown(f"**Investigating:** `{company}` ({i+1}/{len(survivors)})")

            with st.spinner(f"Agent working on {company}…"):
                outcome = investigate(record)

            completeness = score_pair(
                record,
                outcome["result"].get("corrected_record", record),
            )
            st.session_state.results.append(
                {
                    "original": record,
                    "result": outcome["result"],
                    "trace": outcome["trace"],
                    "completeness": completeness,
                }
            )
            progress.progress((i + 1) / len(survivors), text=f"Done {i+1}/{len(survivors)}")

        status.success(
            f"Done — {len(survivors)} records investigated, "
            f"{len(junk_indices)} junk removed, {len(same)} duplicates removed."
        )

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    results = st.session_state.results

    n_total = len(results)
    n_junk = sum(1 for r in results if r["result"].get("is_junk"))
    n_flagged = sum(1 for r in results if r["result"].get("flags"))
    avg_before = round(sum(r["completeness"]["before"]["score_pct"] for r in results) / n_total)
    avg_after = round(sum(r["completeness"]["after"]["score_pct"] for r in results) / n_total)
    cache_stats = search_cache.stats()

    st.subheader(f"Results — {n_total} records")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Investigated", n_total)
    m2.metric("Junk / test", n_junk)
    m3.metric("Flagged fields", n_flagged)
    m4.metric("Cache hits", cache_stats["hits"])
    m5.metric("Avg completeness before", f"{avg_before}%")
    m6.metric("Avg completeness after", f"{avg_after}%", delta=f"+{avg_after - avg_before}%")

    tab_records, tab_download = st.tabs(["📋 Record Detail", "⬇️ Download"])

    with tab_records:
        for i, item in enumerate(results):
            original = item["original"]
            result = item["result"]
            trace = item["trace"]
            comp = item["completeness"]
            company = original.get("company_name", f"Record {i+1}")
            conf = result.get("confidence", "low")
            tool_calls = [t for t in trace if t["type"] == "tool_call"]
            is_junk = result.get("is_junk", False)

            badges = _confidence_badge(conf)
            if is_junk:
                badges += ' <span class="badge-junk">JUNK</span>'

            header = (
                f"**{company}** — "
                f"{len(result.get('changes', []))} change(s) · "
                f"{len(tool_calls)} tool call(s) · " + badges
            )

            with st.expander(header, expanded=(i == 0)):
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**Completeness before**")
                    st.markdown(_completeness_bar(comp["before"]["score_pct"]), unsafe_allow_html=True)
                    if comp["before"]["missing"]:
                        st.caption("Missing: " + ", ".join(comp["before"]["missing"]))
                with col_a:
                    st.markdown("**Completeness after**")
                    st.markdown(_completeness_bar(comp["after"]["score_pct"]), unsafe_allow_html=True)
                    if comp["after"]["missing"]:
                        st.caption("Still missing: " + ", ".join(comp["after"]["missing"]))

                st.divider()

                if trace:
                    st.markdown("**Agent reasoning & tool calls**")
                    _render_trace(trace)
                    st.divider()

                col_c, col_d = st.columns(2)
                with col_c:
                    st.markdown("**Changes made**")
                    for c in result.get("changes", []) or ["No changes."]:
                        st.markdown(f"- {c}")
                with col_d:
                    st.markdown("**Flagged for review**")
                    flags = result.get("flags", [])
                    if flags:
                        for f in flags:
                            st.warning(f)
                    else:
                        st.caption("Nothing flagged.")

                st.markdown("**Before → After**")
                _render_diff(original, result.get("corrected_record", original))

    with tab_download:
        out_df = _build_output_df(results)
        st.dataframe(out_df, use_container_width=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name=f"mdm_agent_output_{ts}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download JSON",
            data=out_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
            file_name=f"mdm_agent_output_{ts}.json",
            mime="application/json",
        )
