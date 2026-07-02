"""Streamlit UI for the autonomous agent: Approval Queue, Run Agent trigger,
and the Performance dashboard.

Kept separate from ar_automation.py (the original, untouched manual-dashboard
module) and from ar_agent.py (pure backend agent logic) — this file is only
the new UI surface that plugs those two together.
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ar_automation import (
    ACCENT,
    BG_CHART,
    BG_PAPER,
    DANGER,
    NEUTRAL,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    _get_secret,
    _TIER_COLORS,
    get_recent_logs,
)
from ar_agent import (
    _send_via_sendgrid,
    get_agent_run_history,
    get_all_invoices,
    get_last_agent_run,
    get_pending_approval,
    run_ar_agent,
    update_invoice,
)
from ar_automation import log_action

_TIER_ORDER = ["Current", "At Risk", "Overdue", "Critical"]
_FONT = "Inter, Arial, sans-serif"

_DECISION_COLORS = {
    "auto_sent":        SUCCESS,
    "approved_sent":    ACCENT,
    "pending_approval": WARNING,
    "monitored":        NEUTRAL,
    "human_override":   DANGER,
    "send_failed":      "#6D597A",
}
_DECISION_LABELS = {
    "auto_sent":        "Auto Sent",
    "approved_sent":    "Approved & Sent",
    "pending_approval": "Pending Approval",
    "monitored":        "Monitored",
    "human_override":   "Human Override",
    "send_failed":      "Send Failed",
}


def _chart_layout(title: str, **extra) -> dict:
    layout = dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(family=_FONT, size=14, color=TEXT_PRIMARY), x=0),
        plot_bgcolor=BG_CHART,
        paper_bgcolor=BG_PAPER,
        font=dict(family=_FONT, color=TEXT_SECONDARY),
        height=340,
        margin=dict(t=40, b=30, l=40, r=10),
    )
    layout.update(extra)
    return layout


# ---------------------------------------------------------------------------
# Addition 2: Approval Queue
# ---------------------------------------------------------------------------

def pending_approval_count() -> int:
    try:
        return len(get_pending_approval())
    except Exception:
        return 0


def render_approval_queue() -> None:
    st.subheader("Approval Queue")
    st.caption("Invoices the agent held for human review — over $25,000, or Critical risk tier.")

    df = get_pending_approval()
    if df.empty:
        st.info("Nothing waiting for approval. Run the agent from the \"Run Agent\" tab.")
        return

    for _, inv in df.iterrows():
        draft = json.loads(inv["draft_email_json"]) if inv["draft_email_json"] else {}
        with st.expander(
            f"{inv['invoice_number']} — {inv['client_name']} — "
            f"${inv['amount_due']:,.2f} ({inv['risk_tier']})",
            expanded=False,
        ):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Amount Due", f"${inv['amount_due']:,.2f}")
            d2.metric("Days Overdue", f"{int(inv['days_overdue'])}")
            d3.metric("Risk Tier", inv["risk_tier"])
            d4.metric("Due Date", str(inv["due_date"]))

            st.markdown(f"**Subject:** {draft.get('subject_line', inv['email_subject'] or '')}")
            st.text_area("Email Body", draft.get("email_body", ""), height=180,
                         disabled=True, key=f"body_{inv['invoice_number']}")
            st.caption(
                f"Tone: {draft.get('tone', '—')}  |  Urgency: {draft.get('urgency_level', '—')}  |  "
                f"Cited policy: {', '.join(draft.get('cited_policy', [])) or '—'}"
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve and Send", type="primary",
                             key=f"approve_{inv['invoice_number']}"):
                    sender_email = _get_secret("SENDER_EMAIL")
                    try:
                        message_id = _send_via_sendgrid(
                            inv["client_email"], draft.get("subject_line", ""),
                            draft.get("email_body", ""), sender_email,
                        )
                        update_invoice(inv["invoice_number"], decision="approved_sent",
                                      sendgrid_message_id=message_id)
                        log_action("Reviewer", "approved_sent", invoice_number=inv["invoice_number"],
                                  details={"client_name": inv["client_name"],
                                           "sendgrid_message_id": message_id})
                        st.success("Sent.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"SendGrid send failed: {e}")
            with c2:
                reason = st.text_input("Override reason", key=f"reason_{inv['invoice_number']}")
                if st.button("Override — Do Not Send", key=f"override_{inv['invoice_number']}"):
                    if not reason.strip():
                        st.warning("Please provide a reason before overriding.")
                    else:
                        update_invoice(inv["invoice_number"], decision="human_override",
                                      override_reason=reason.strip())
                        log_action("Reviewer", "human_override", invoice_number=inv["invoice_number"],
                                  details={"client_name": inv["client_name"], "reason": reason.strip()})
                        st.success("Marked as overridden — not sent.")
                        st.rerun()


# ---------------------------------------------------------------------------
# Addition 3: Run Agent
# ---------------------------------------------------------------------------

def render_agent_trigger() -> None:
    st.subheader("Run Agent")
    st.caption(
        "Runs the autonomous agent over every open invoice: auto-sends low-risk "
        "reminders, holds high-risk/high-value invoices for approval."
    )

    demo_mode = st.toggle(
        "Demo Mode — send all auto-approved emails to a single test address "
        "(st.secrets['DEMO_EMAIL']) instead of real client addresses",
        value=True,
    )

    last_run = get_last_agent_run()
    if last_run:
        st.caption(f"Last run: {last_run['run_at']}")
    else:
        st.caption("Agent has not been run yet.")

    if st.button("Run AR Agent Now", type="primary"):
        with st.spinner("Running AR agent..."):
            summary = run_ar_agent(demo_mode=demo_mode)
        st.success("Agent run complete.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Processed", summary["total_processed"])
        c2.metric("Auto Sent", summary["auto_sent"])
        c3.metric("Pending Approval", summary["pending_approval"])
        c4.metric("Monitored", summary["monitored"])
        c5.metric("Errors", summary["errors"])
        st.rerun()


# ---------------------------------------------------------------------------
# Addition 4: Performance dashboard
# ---------------------------------------------------------------------------

def _chart_tier_distribution(df: pd.DataFrame) -> go.Figure:
    agg = (df.groupby("risk_tier")
           .agg(count=("invoice_number", "count"), total=("amount_due", "sum"))
           .reindex(_TIER_ORDER)
           .fillna(0)
           .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["risk_tier"], y=agg["count"], name="Invoice Count",
        marker_color=[_TIER_COLORS.get(t, NEUTRAL) for t in agg["risk_tier"]],
        yaxis="y1", width=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=agg["risk_tier"], y=agg["total"], name="Total $ Amount",
        yaxis="y2", mode="lines+markers",
        line=dict(color=TEXT_PRIMARY, width=2, dash="dot"),
        marker=dict(size=9, color=TEXT_PRIMARY, line=dict(color=BG_CHART, width=1)),
    ))
    layout = _chart_layout(
        "Invoice Distribution by Risk Tier",
        yaxis=dict(title="Invoice Count", tickfont=dict(color=TEXT_SECONDARY),
                  rangemode="tozero", gridcolor="rgba(141,153,174,0.15)"),
        yaxis2=dict(title="Total $ Amount", overlaying="y", side="right",
                    tickfont=dict(color=TEXT_SECONDARY), rangemode="tozero",
                    showgrid=False, tickprefix="$"),
        showlegend=True, bargap=0.35,
        legend=dict(orientation="h", y=1.18, x=0,
                    font=dict(color=TEXT_SECONDARY, size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, b=30, l=50, r=50),
    )
    fig.update_layout(**layout)
    return fig


def _chart_aging_funnel(df: pd.DataFrame) -> go.Figure:
    agg = (df.groupby("risk_tier")
           .agg(count=("invoice_number", "count"), total=("amount_due", "sum"))
           .reindex(_TIER_ORDER)
           .fillna(0)
           .reset_index())

    fig = go.Figure(go.Funnel(
        y=agg["risk_tier"],
        x=agg["count"],
        textinfo="text",
        text=[f"{int(c)} invoices — ${t:,.0f}" for c, t in zip(agg["count"], agg["total"])],
        textfont=dict(color=BG_PAPER, family=_FONT, size=13),
        connector=dict(fillcolor="rgba(141,153,174,0.15)"),
        marker=dict(color=[_TIER_COLORS.get(t, NEUTRAL) for t in agg["risk_tier"]]),
    ))
    fig.update_layout(**_chart_layout(
        "AR Aging Funnel", height=420,
        yaxis=dict(tickfont=dict(color=TEXT_PRIMARY, size=13)),
        margin=dict(t=50, b=20, l=100, r=40),
    ))
    return fig


def _chart_decision_breakdown(df: pd.DataFrame) -> go.Figure:
    counts = df["decision"].fillna("unprocessed").value_counts().reset_index()
    counts.columns = ["decision", "count"]
    counts["label"] = counts["decision"].map(_DECISION_LABELS).fillna(counts["decision"])
    colors = [_DECISION_COLORS.get(d, NEUTRAL) for d in counts["decision"]]

    fig = px.pie(counts, names="label", values="count", color="label",
                color_discrete_sequence=colors, hole=0.45)
    fig.update_traces(
        textfont=dict(color=TEXT_PRIMARY, family=_FONT, size=12),
        texttemplate="%{label}<br>%{value} (%{percent})",
        marker=dict(line=dict(color=BG_PAPER, width=2)),
    )
    fig.update_layout(**_chart_layout("Agent Decision Breakdown", showlegend=False,
                                      margin=dict(t=50, b=20, l=20, r=20)))
    return fig


def _chart_aging_trend():
    logs = get_recent_logs(limit=2000)
    rows = []
    for entry in logs:
        if entry.get("role") != "AR Agent":
            continue
        details = entry.get("details") or {}
        tier = details.get("risk_tier")
        if not tier:
            continue
        date = entry["timestamp"][:10]
        rows.append({"date": date, "risk_tier": tier})

    if not rows:
        st.info("Run the agent on multiple days to see trends.")
        return

    trend_df = pd.DataFrame(rows)
    distinct_dates = trend_df["date"].nunique()
    if distinct_dates < 3:
        st.info("Run the agent on multiple days to see trends.")
        return

    agg = trend_df.groupby(["date", "risk_tier"]).size().reset_index(name="count")
    fig = px.line(agg, x="date", y="count", color="risk_tier",
                 color_discrete_map=_TIER_COLORS, markers=True)
    fig.update_layout(**_chart_layout("Invoice Aging Over Time", showlegend=True,
                                      legend=dict(font=dict(color=TEXT_SECONDARY, size=10),
                                                  bgcolor="rgba(0,0,0,0)")))
    st.plotly_chart(fig, use_container_width=True)


def render_performance_dashboard() -> None:
    st.subheader("Performance")

    invoices = get_all_invoices()
    runs = get_agent_run_history()

    if invoices.empty:
        st.info("Upload invoices and run the agent to see performance data.")
        return

    total_processed = int(runs["total_processed"].sum()) if not runs.empty else 0
    total_sent = (
        (int(runs["auto_sent"].sum()) if not runs.empty else 0)
        + int((invoices["decision"] == "approved_sent").sum())
    )
    total_outstanding = float(invoices["amount_due"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Processed (to date)", f"{total_processed:,}")
    c2.metric("Total Emails Sent", f"{total_sent:,}")
    c3.metric("Total Outstanding", f"${total_outstanding:,.2f}")

    st.divider()

    # Hero chart — the aging funnel is the single most visually impactful view,
    # so it gets its own full-width row instead of competing for space.
    st.plotly_chart(_chart_aging_funnel(invoices), use_container_width=True)

    st.divider()

    r1c1, r1c2 = st.columns(2)
    r1c1.plotly_chart(_chart_tier_distribution(invoices), use_container_width=True)
    r1c2.plotly_chart(_chart_decision_breakdown(invoices), use_container_width=True)

    st.divider()
    _chart_aging_trend()
