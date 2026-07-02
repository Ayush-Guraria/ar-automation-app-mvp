"""AR Automation — invoice risk classification and RAG-grounded collections email drafting.

Fully self-contained Streamlit app module: owns its Claude client config, its
own ChromaDB policy index, its own JSON-parsing helpers, and its own SQLite
audit trail. Has no dependency on any other project.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

_APP_ROOT = Path(__file__).resolve().parent
_CHROMA_PATH = _APP_ROOT / "chroma_db"
_POLICIES_DIR = _APP_ROOT / "policies"
_AUDIT_DB_PATH = _APP_ROOT / "ar_audit.db"

COLLECTION_NAME = "ar_policies"
_CHUNK_SIZE = 1200  # ~300 tokens at ~4 chars/token
_OVERLAP = 200      # ~50 tokens

# ---------------------------------------------------------------------------
# Color constants — mirror the anomaly-detection app's dark-theme palette so
# both apps feel like part of the same portfolio.
# ---------------------------------------------------------------------------
ACCENT         = "#00B4D8"
DANGER         = "#E63946"
WARNING        = "#F4A261"
SUCCESS        = "#2A9D8F"
NEUTRAL        = "#8D99AE"
TEXT_PRIMARY   = "#FFFFFF"
TEXT_SECONDARY = "#ADB5BD"
BG_CHART       = "#1A1A2E"
BG_PAPER       = "#16213E"

REQUIRED_COLUMNS = ["invoice_number", "client_name", "amount_due", "due_date", "days_overdue"]

_TIER_COLORS = {
    "Current":  SUCCESS,
    "At Risk":  "#E9C46A",
    "Overdue":  WARNING,
    "Critical": DANGER,
}

_EXPECTED_EMAIL_KEYS = ("subject_line", "email_body", "tone", "cited_policy", "urgency_level")


# ---------------------------------------------------------------------------
# Config — Claude API key/model from st.secrets first, falling back to .env
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str | None = None) -> str | None:
    """Look up a config value in st.secrets, then os.environ, then default."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def get_anthropic_api_key() -> str:
    load_dotenv(_APP_ROOT / ".env", override=False)
    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to .streamlit/secrets.toml "
            "(Streamlit Cloud) or a local .env file."
        )
    return api_key


def get_claude_model() -> str:
    return _get_secret("CLAUDE_MODEL", "claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# RAG index — own ChromaDB instance, own collection, own policy docs
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str, source: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _OVERLAP
) -> list[dict]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"id": f"{source}_{idx}", "text": text[start:end], "source": source})
        idx += 1
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def _load_policies(policies_dir: Path) -> list[tuple[str, str]]:
    """Read all .md files from the policies directory, sorted by name."""
    return [
        (f.name, f.read_text(encoding="utf-8"))
        for f in sorted(policies_dir.glob("*.md"))
    ]


def _get_collection(chroma_path: Path, *, create: bool):
    """Return a chromadb collection, optionally dropping and recreating it."""
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(chroma_path))
    if create:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(COLLECTION_NAME, embedding_function=ef)
    return client.get_collection(COLLECTION_NAME, embedding_function=ef)


def build_index() -> None:
    """Build (or rebuild) the AR policy ChromaDB collection.

    Idempotent: drops the existing collection before recreating it.
    """
    _CHROMA_PATH.mkdir(exist_ok=True)
    collection = _get_collection(_CHROMA_PATH, create=True)
    policies = _load_policies(_POLICIES_DIR)

    all_chunks: list[dict] = []
    for filename, content in policies:
        all_chunks.extend(_chunk_text(content, filename))

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Query the AR policy index and return the top-k chunks."""
    collection = _get_collection(_CHROMA_PATH, create=False)
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove leading ```json / ``` and trailing ``` wrappers."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def _parse_json(text: str) -> dict | None:
    for candidate in (text, _strip_fences(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Audit log — own SQLite database
# ---------------------------------------------------------------------------

def _init_audit_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ar_audit_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                role           TEXT NOT NULL,
                action         TEXT NOT NULL,
                invoice_number TEXT,
                details        TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_action(
    role: str,
    action: str,
    invoice_number: str | None = None,
    details: dict | None = None,
) -> None:
    """Insert one audit row. details is serialised to JSON."""
    _init_audit_db(_AUDIT_DB_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()
    details_json = json.dumps(details) if details is not None else None
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO ar_audit_log (timestamp, role, action, invoice_number, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, role, action, invoice_number, details_json),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_logs(limit: int = 50) -> list[dict]:
    """Return up to *limit* audit rows, newest first."""
    _init_audit_db(_AUDIT_DB_PATH)
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, timestamp, role, action, invoice_number, details "
            "FROM ar_audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        entry = dict(row)
        if entry["details"] is not None:
            try:
                entry["details"] = json.loads(entry["details"])
            except json.JSONDecodeError:
                pass
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def _classify_risk_tier(days_overdue: int) -> str:
    """Rule-based risk tier from days overdue."""
    if days_overdue <= 30:
        return "Current"
    if days_overdue <= 60:
        return "At Risk"
    if days_overdue <= 90:
        return "Overdue"
    return "Critical"


def _validate_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of missing required columns (empty if valid)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _classify_invoices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_tier"] = out["days_overdue"].astype(int).map(_classify_risk_tier)
    return out


def _style_risk_tier(df: pd.DataFrame):
    """Return a pandas Styler that color-codes the risk_tier column."""
    def _bg(val: str) -> str:
        color = _TIER_COLORS.get(val, NEUTRAL)
        return f"background-color: {color}; color: #16213E; font-weight: 600;"
    return (df.style
            .map(_bg, subset=["risk_tier"])
            .format({"amount_due": "${:,.2f}"}))


# ---------------------------------------------------------------------------
# RAG + Claude email drafting
# ---------------------------------------------------------------------------

def _ar_query() -> str:
    return "accounts receivable payment terms follow-up policy"


def _format_chunks(chunks: list[dict]) -> str:
    parts = [
        f"[Policy Chunk {i} — {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n".join(parts)


def _ar_system_message() -> str:
    return (
        "You are an AR collections assistant at a financial institution. "
        "You draft follow-up emails for overdue invoices, grounded in the internal "
        "AR payment-terms policy. Always respond with valid JSON only — no prose, "
        "no markdown code fences."
    )


def _ar_user_message(invoice: dict, chunks: list[dict]) -> str:
    inv_lines = "\n".join(f"  {k}: {v}" for k, v in invoice.items())
    policy_text = _format_chunks(chunks)
    return (
        f"Invoice details:\n{inv_lines}\n\n"
        f"Relevant AR policy excerpts:\n{policy_text}\n\n"
        "Calibrate tone to the invoice's risk_tier:\n"
        "- Current: professional and gentle payment reminder\n"
        "- At Risk: firm reminder citing payment terms and any applicable late fee\n"
        "- Overdue: urgent follow-up requesting immediate payment\n"
        "- Critical: escalation language, note that management is being CC'd\n\n"
        "Return a JSON object with exactly these keys:\n"
        '- "subject_line": short email subject line\n'
        '- "email_body": full email body text, citing the specific policy language above\n'
        '- "tone": one of "gentle", "firm", "urgent", "escalation"\n'
        '- "cited_policy": list of policy filenames/sections cited\n'
        '- "urgency_level": one of "low", "medium", "high", "critical"\n'
        "JSON only — no other text."
    )


def _call_claude(client: anthropic.Anthropic, model: str, system: str, messages: list[dict]) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=768,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def draft_followup_email(invoice: dict) -> dict:
    """Return a structured follow-up email draft for one invoice.

    Calls Claude with RAG-retrieved AR policy context.
    """
    client = anthropic.Anthropic(api_key=get_anthropic_api_key())
    model = get_claude_model()

    chunks = retrieve(_ar_query(), k=3)
    system = _ar_system_message()
    messages = [{"role": "user", "content": _ar_user_message(invoice, chunks)}]

    raw = _call_claude(client, model, system, messages)
    result = _parse_json(raw)

    if result is None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": "Respond with valid JSON only."})
        raw = _call_claude(client, model, system, messages)
        result = _parse_json(raw)

    if result is None:
        result = {
            "subject_line": f"Follow-up: Invoice {invoice.get('invoice_number', '')}",
            "email_body": raw,
            "tone": "firm",
            "cited_policy": [],
            "urgency_level": "medium",
        }

    return result


# ---------------------------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------------------------

def _metric_row(df: pd.DataFrame) -> None:
    total_outstanding = df["amount_due"].sum()
    avg_days_overdue = df["days_overdue"].mean()
    critical_amount = df.loc[df["risk_tier"] == "Critical", "amount_due"].sum()

    tier_counts = df["risk_tier"].value_counts()
    breakdown = " | ".join(
        f"{tier}: {tier_counts.get(tier, 0)}"
        for tier in ["Current", "At Risk", "Overdue", "Critical"]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Outstanding", f"${total_outstanding:,.2f}")
    c2.metric("Invoices by Risk Tier", f"{len(df):,}", breakdown)
    c3.metric("Average Days Overdue", f"{avg_days_overdue:.1f}")
    c4.metric("Total Critical Amount", f"${critical_amount:,.2f}")


# ---------------------------------------------------------------------------
# Email draft panel
# ---------------------------------------------------------------------------

def _show_email_draft(result: dict) -> None:
    st.markdown(f"**Subject:** {result.get('subject_line', '')}")
    st.text_area("Email Body", result.get("email_body", ""), height=220, disabled=True)

    urgency = str(result.get("urgency_level", "low")).lower()
    tone = result.get("tone", "")
    if urgency == "critical":
        st.error(f"Urgency: Critical  |  Tone: {tone}")
    elif urgency == "high":
        st.warning(f"Urgency: High  |  Tone: {tone}")
    else:
        st.success(f"Urgency: {urgency.title()}  |  Tone: {tone}")

    cited = result.get("cited_policy", [])
    if cited:
        st.markdown("**Cited Policy**")
        for ref in cited:
            st.markdown(f"- {ref}")


def _invoice_detail_panel(df: pd.DataFrame) -> None:
    inv_id = st.session_state.ar_selected_invoice
    if inv_id is None:
        return

    mask = df["invoice_number"].astype(str) == inv_id
    if not mask.any():
        st.warning(f"Invoice {inv_id} not found.")
        return
    row = df[mask].iloc[0]

    with st.expander(f"Invoice {inv_id} — {row['client_name']}", expanded=True):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Amount Due", f"${row['amount_due']:,.2f}")
        d2.metric("Days Overdue", f"{int(row['days_overdue'])}")
        d3.metric("Risk Tier", row["risk_tier"])
        d4.metric("Due Date", str(row["due_date"]))

        cache = st.session_state.ar_email_cache
        if inv_id in cache:
            st.caption("(cached — no API call)")
            _show_email_draft(cache[inv_id])
        elif st.button("Draft Follow-Up Email", type="primary", key=f"draft_{inv_id}"):
            invoice = row.to_dict()
            with st.spinner("Asking Claude..."):
                result = draft_followup_email(invoice)
            cache[inv_id] = result
            log_action(
                "AR Automation", "email_drafted", invoice_number=inv_id,
                details={
                    "client_name": row["client_name"],
                    "risk_tier": row["risk_tier"],
                    "subject_line": result.get("subject_line", ""),
                },
            )
            _show_email_draft(result)

        if st.button("Clear Selection", key=f"clear_{inv_id}"):
            st.session_state.ar_selected_invoice = None
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_ar_app() -> None:
    """Render the full AR Automation app body."""
    for key, default in {
        "ar_df": None,
        "ar_selected_invoice": None,
        "ar_email_cache": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    uploaded = st.file_uploader("Upload AR invoice CSV", type=["csv"], key="ar_uploader")
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        missing = _validate_columns(raw_df)
        if missing:
            st.error(f"CSV is missing required column(s): {', '.join(missing)}")
            return
        st.session_state.ar_df = _classify_invoices(raw_df)
        st.session_state.ar_email_cache = {}
        st.session_state.ar_selected_invoice = None

    df = st.session_state.ar_df
    if df is None:
        st.info("Upload a CSV to get started, or run `python generate_ar_data.py` for sample data.")
        return

    st.markdown("#### Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.divider()
    _metric_row(df)

    st.divider()
    st.markdown(f"#### Classified Invoices ({len(df):,})")
    st.caption("👆 Select a row to draft a follow-up email for that invoice.")
    event = st.dataframe(_style_risk_tier(df), use_container_width=True, height=340,
                         hide_index=True, on_select="rerun",
                         selection_mode="single-row", key="ar_table")
    rows = event.selection.rows if hasattr(event, "selection") else []
    if rows:
        inv_id = str(df.iloc[rows[0]]["invoice_number"])
        if inv_id != st.session_state.ar_selected_invoice:
            st.session_state.ar_selected_invoice = inv_id
            st.rerun()

    _invoice_detail_panel(df)

    st.divider()
    st.download_button(
        "Download Classified Invoices (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="classified_ar_invoices.csv",
        mime="text/csv",
    )
