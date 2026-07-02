"""Synthetic AR demo-data generator.

Run once before ``streamlit run app.py`` to produce a sample accounts-receivable
CSV so the AR Automation app has realistic data to demo against. Also seeds the
app's own ChromaDB policy index from policies/ar_receivables_policy.md.

Usage:
    python generate_ar_data.py
"""

import random
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ar_automation import build_index

_APP_ROOT = Path(__file__).resolve().parent
_CSV_PATH = _APP_ROOT / "sample_ar_invoices.csv"

# (tier label, min days overdue, max days overdue)
_TIERS = [
    ("Current", 0, 30),
    ("At Risk", 31, 60),
    ("Overdue", 61, 90),
    ("Critical", 91, 180),
]
_INVOICES_PER_TIER = 5  # 4 tiers x 5 = 20 invoices

_FICTIONAL_CLIENTS = [
    "Bluewater Logistics Co.", "Northfield Office Supply", "Cascade Ridge Builders",
    "Ironvale Manufacturing", "Solstice Retail Group", "Amberton Consulting LLC",
    "Prairie Bell Foods", "Copperline Industrial", "Wrenhollow Media",
    "Granite Creek Distributors", "Vantage Point Analytics", "Marlowe & Finch Design",
    "Silverbrook Textiles", "Eastgate Hardware", "Fernway Dental Partners",
    "Oakmoor Freight Systems", "Thornbury Insurance Group", "Redwood Peak Realty",
    "Meridian Bay Shipping", "Clearview Software Partners",
]

_CONTACT_NAMES = [
    "billing", "accounts", "ap", "finance", "invoices",
]


def _slugify_domain(client_name: str) -> str:
    """Derive a plausible email domain from a fictional client name."""
    core = re.sub(r"[^a-z0-9 ]", "", client_name.lower())
    words = [w for w in core.split() if w not in {"co", "llc", "group", "inc"}]
    return "".join(words[:2]) + ".com"


def _client_email(client_name: str, rng: random.Random) -> str:
    contact = rng.choice(_CONTACT_NAMES)
    return f"{contact}@{_slugify_domain(client_name)}"


def _random_due_date(days_overdue: int, today: date) -> date:
    return today - timedelta(days=days_overdue)


def _generate_invoices() -> pd.DataFrame:
    today = date.today()
    rng = random.Random()
    clients = _FICTIONAL_CLIENTS.copy()
    rng.shuffle(clients)

    rows = []
    invoice_num = 1001
    client_idx = 0
    for tier_label, lo, hi in _TIERS:
        for _ in range(_INVOICES_PER_TIER):
            days_overdue = rng.randint(lo, hi)
            due_date = _random_due_date(days_overdue, today)
            invoice_date = due_date - timedelta(days=30)
            amount_due = round(rng.uniform(500, 50000), 2)
            client_name = clients[client_idx % len(clients)]
            client_idx += 1
            rows.append({
                "invoice_number": f"INV-{invoice_num}",
                "client_name": client_name,
                "client_email": _client_email(client_name, rng),
                "amount_due": amount_due,
                "invoice_date": invoice_date.isoformat(),
                "due_date": due_date.isoformat(),
                "days_overdue": days_overdue,
            })
            invoice_num += 1

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=None).reset_index(drop=True)


def main() -> None:
    df = _generate_invoices()
    df.to_csv(_CSV_PATH, index=False)
    print(f"Wrote {len(df)} synthetic invoices to {_CSV_PATH}")

    build_index()
    print("Seeded ChromaDB policy index from policies/ar_receivables_policy.md.")


if __name__ == "__main__":
    main()
