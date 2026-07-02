"""AR Automation — Streamlit entry point."""

import streamlit as st

from ar_agent import upsert_invoices
from ar_automation import (
    ACCENT,
    BG_PAPER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    build_index,
    render_ar_app,
)
from ar_dashboard import (
    pending_approval_count,
    render_agent_trigger,
    render_approval_queue,
    render_performance_dashboard,
)

_CSS = f"""
<style>
[data-testid="stMetric"] {{
    background-color: {BG_PAPER};
    border: 1px solid {ACCENT};
    border-radius: 8px;
    padding: 16px;
}}
[data-testid="stMetric"] label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 24px !important;
    font-weight: 700;
}}
[data-testid="stMetricDelta"] {{
    font-size: 11px !important;
}}
[data-testid="stSidebar"] {{
    background-color: {BG_PAPER};
    border-right: 1px solid {ACCENT};
}}
[data-testid="stSidebar"] label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
</style>
"""


@st.cache_resource
def _init_rag_index() -> None:
    """Build the ChromaDB policy index once per process lifetime.

    Runs exactly once on cold start; skipped on every subsequent Streamlit
    rerun. On Streamlit Community Cloud the filesystem is ephemeral, so
    every cold start triggers a fresh build from the committed policy docs.
    """
    build_index()


def main() -> None:
    st.set_page_config(page_title="AR Automation", layout="wide", page_icon="🧾")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("AR Automation")
    st.caption(
        "Upload an accounts-receivable CSV to classify invoices by overdue risk, "
        "then draft a policy-grounded follow-up email for any invoice — powered by "
        "Claude and grounded in your internal AR payment-terms policy via RAG."
    )

    _init_rag_index()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧾 AR Automation",
        f"📥 Approval Queue ({pending_approval_count()})",
        "🤖 Run Agent",
        "📊 Performance",
    ])
    with tab1:
        render_ar_app()
        if st.session_state.get("ar_df") is not None:
            upsert_invoices(st.session_state.ar_df)
    with tab2:
        render_approval_queue()
    with tab3:
        render_agent_trigger()
    with tab4:
        render_performance_dashboard()


if __name__ == "__main__":
    main()
