"""Autonomous AR processing agent.

Walks every open invoice in the persistent invoice registry and makes an
autonomous decision per risk tier and dollar amount:

- Current (0-30 days): no action, logged as "monitored".
- At Risk (31-60 days): auto-drafted gentle reminder, sent via SendGrid.
- Overdue (61-90 days): auto-drafted firm follow-up, sent via SendGrid.
- Critical (90+ days): drafted but held for human approval.
- Any invoice over $25,000, regardless of tier: held for human approval.

Reuses ar_automation.py's Claude/RAG drafting (draft_followup_email), audit
logging (log_action), and SQLite database file unchanged. Adds two new
tables (ar_invoices, ar_agent_runs) to that same database for invoice state
and run history — ar_automation.py's own audit_log table is untouched.

Callable both from the Streamlit UI (via run_ar_agent()) and standalone for
scheduled/cron runs: `python ar_agent.py`.
"""

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ar_automation import (
    _AUDIT_DB_PATH,
    _classify_risk_tier,
    _get_secret,
    draft_followup_email,
    log_action,
)

_HIGH_VALUE_THRESHOLD = 25_000
_OPEN_DECISIONS = ("monitored", "send_failed")

_INVOICE_COLUMNS = [
    "invoice_number", "client_name", "client_email", "amount_due",
    "invoice_date", "due_date", "days_overdue", "risk_tier",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _init_tables() -> None:
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ar_invoices (
                invoice_number      TEXT PRIMARY KEY,
                client_name         TEXT,
                client_email        TEXT,
                amount_due          REAL,
                invoice_date        TEXT,
                due_date            TEXT,
                days_overdue        INTEGER,
                risk_tier           TEXT,
                decision            TEXT,
                email_subject       TEXT,
                draft_email_json    TEXT,
                sendgrid_message_id TEXT,
                override_reason     TEXT,
                updated_at          TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ar_agent_runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at           TEXT NOT NULL,
                demo_mode        INTEGER NOT NULL,
                total_processed  INTEGER NOT NULL,
                auto_sent        INTEGER NOT NULL,
                pending_approval INTEGER NOT NULL,
                monitored        INTEGER NOT NULL,
                errors           INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Invoice registry
# ---------------------------------------------------------------------------

def upsert_invoices(df: pd.DataFrame) -> int:
    """Insert any invoice_numbers not already tracked. Existing rows (and

    their decision/progress) are left untouched, so re-uploading the same
    CSV never resets an invoice already handled by the agent.
    Returns the number of new rows inserted.
    """
    _init_tables()
    if "risk_tier" not in df.columns:
        df = df.copy()
        df["risk_tier"] = df["days_overdue"].astype(int).map(_classify_risk_tier)

    conn = sqlite3.connect(_AUDIT_DB_PATH)
    inserted = 0
    try:
        for _, row in df.iterrows():
            values = {col: row.get(col) for col in _INVOICE_COLUMNS}
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO ar_invoices
                    (invoice_number, client_name, client_email, amount_due,
                     invoice_date, due_date, days_overdue, risk_tier, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["invoice_number"], values["client_name"], values.get("client_email"),
                    float(values["amount_due"]), values.get("invoice_date"), values["due_date"],
                    int(values["days_overdue"]), values["risk_tier"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def get_open_invoices() -> list[dict]:
    """Invoices not yet terminally handled (send_failed is retried each run)."""
    _init_tables()
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(_OPEN_DECISIONS))
        rows = conn.execute(
            f"SELECT * FROM ar_invoices WHERE decision IS NULL OR decision IN ({placeholders})",
            _OPEN_DECISIONS,
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_pending_approval() -> pd.DataFrame:
    _init_tables()
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT * FROM ar_invoices WHERE decision = 'pending_approval' ORDER BY amount_due DESC",
            conn,
        )
    finally:
        conn.close()


def get_all_invoices() -> pd.DataFrame:
    _init_tables()
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM ar_invoices", conn)
    finally:
        conn.close()


def update_invoice(invoice_number: str, **fields) -> None:
    if not fields:
        return
    fields = {**fields, "updated_at": datetime.now(timezone.utc).isoformat()}
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        conn.execute(
            f"UPDATE ar_invoices SET {set_clause} WHERE invoice_number = ?",
            (*fields.values(), invoice_number),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Agent run history
# ---------------------------------------------------------------------------

def record_agent_run(demo_mode: bool, summary: dict) -> None:
    _init_tables()
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO ar_agent_runs
                (run_at, demo_mode, total_processed, auto_sent, pending_approval, monitored, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(), int(demo_mode),
                summary["total_processed"], summary["auto_sent"],
                summary["pending_approval"], summary["monitored"], summary["errors"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_last_agent_run() -> dict | None:
    _init_tables()
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM ar_agent_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return _row_to_dict(row) if row else None


def get_agent_run_history() -> pd.DataFrame:
    _init_tables()
    conn = sqlite3.connect(_AUDIT_DB_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM ar_agent_runs ORDER BY id", conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SendGrid
# ---------------------------------------------------------------------------

def _send_via_sendgrid(to_email: str, subject: str, body: str, sender_email: str) -> str:
    """Send one email via SendGrid. Raises RuntimeError if not configured,
    or re-raises the SendGrid client's own exception on API failure.
    """
    api_key = _get_secret("SENDGRID_API_KEY")
    if not api_key or not sender_email or not to_email:
        raise RuntimeError(
            "SendGrid is not fully configured (SENDGRID_API_KEY/SENDER_EMAIL/recipient missing)."
        )

    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    message = Mail(from_email=sender_email, to_emails=to_email,
                    subject=subject, plain_text_content=body)
    response = SendGridAPIClient(api_key).send(message)
    return response.headers.get("X-Message-Id", "")


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def _decide(risk_tier: str, amount_due: float) -> str:
    """Return one of: monitored, pending_approval, auto_send_gentle, auto_send_firm."""
    if risk_tier == "Current":
        base = "monitored"
    elif risk_tier == "At Risk":
        base = "auto_send_gentle"
    elif risk_tier == "Overdue":
        base = "auto_send_firm"
    else:  # Critical
        base = "pending_approval"

    if amount_due > _HIGH_VALUE_THRESHOLD:
        return "pending_approval"
    return base


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_ar_agent(demo_mode: bool = False) -> dict:
    """Process every open invoice and return a summary of decisions taken."""
    _init_tables()
    summary = {"total_processed": 0, "auto_sent": 0, "pending_approval": 0,
               "monitored": 0, "errors": 0}

    sender_email = _get_secret("SENDER_EMAIL")
    demo_email = _get_secret("DEMO_EMAIL")

    for inv in get_open_invoices():
        inv_num = inv["invoice_number"]
        try:
            action = _decide(inv["risk_tier"], inv["amount_due"])

            if action == "monitored":
                update_invoice(inv_num, decision="monitored")
                log_action(
                    "AR Agent", "monitored", invoice_number=inv_num,
                    details={"client_name": inv["client_name"], "risk_tier": inv["risk_tier"],
                             "amount_due": inv["amount_due"]},
                )
                summary["monitored"] += 1

            elif action == "pending_approval":
                draft = draft_followup_email(inv)
                update_invoice(
                    inv_num, decision="pending_approval",
                    email_subject=draft.get("subject_line"),
                    draft_email_json=json.dumps(draft),
                )
                log_action(
                    "AR Agent", "pending_approval", invoice_number=inv_num,
                    details={"client_name": inv["client_name"], "risk_tier": inv["risk_tier"],
                             "amount_due": inv["amount_due"],
                             "email_subject": draft.get("subject_line")},
                )
                summary["pending_approval"] += 1

            else:  # auto_send_gentle / auto_send_firm
                draft = draft_followup_email(inv)
                to_email = demo_email if demo_mode else inv.get("client_email")
                try:
                    message_id = _send_via_sendgrid(
                        to_email, draft.get("subject_line", ""), draft.get("email_body", ""),
                        sender_email,
                    )
                    update_invoice(
                        inv_num, decision="auto_sent",
                        email_subject=draft.get("subject_line"),
                        draft_email_json=json.dumps(draft),
                        sendgrid_message_id=message_id,
                    )
                    log_action(
                        "AR Agent", "auto_sent", invoice_number=inv_num,
                        details={"client_name": inv["client_name"], "risk_tier": inv["risk_tier"],
                                 "amount_due": inv["amount_due"],
                                 "email_subject": draft.get("subject_line"),
                                 "sendgrid_message_id": message_id, "demo_mode": demo_mode},
                    )
                    summary["auto_sent"] += 1
                except Exception as send_err:
                    update_invoice(
                        inv_num, decision="send_failed",
                        email_subject=draft.get("subject_line"),
                        draft_email_json=json.dumps(draft),
                    )
                    log_action(
                        "AR Agent", "send_error", invoice_number=inv_num,
                        details={"error": str(send_err), "risk_tier": inv["risk_tier"]},
                    )
                    summary["errors"] += 1

            summary["total_processed"] += 1
        except Exception as e:
            log_action("AR Agent", "processing_error", invoice_number=inv_num,
                       details={"error": str(e)})
            summary["errors"] += 1

    record_agent_run(demo_mode, summary)
    return summary


if __name__ == "__main__":
    result = run_ar_agent(demo_mode=False)
    print(json.dumps(result, indent=2))
