"""
The 90-Day Safety Check.

Projects a user's balance forward from `request_date` using:
  - recurring expenses/income (detected patterns, projected at their interval)
  - confirmed one-off future events already in financial_events.csv
    (status in {scheduled, pending} and event_date/settlement_date in window)
  - any extra candidate payment(s) we're testing (e.g. "pay 5000 on day 0")
  - optional spending changes (stop / reduce_to) applied to specific
    recurring occurrences before summing

A plan is SAFE only if balance never drops below `minimum_balance_to_keep`
on any day of the 90-day window.

NOTE on double counting: if a recurring pattern's detected occurrences
include events that are also individually scheduled/pending in the window,
we would double count them. We guard against this by excluding, from the
recurring projection, any projected date that falls within
`dedup_window_days` of an already-present real event of the same
(category, direction) in the window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from events import (
    CASH_SAFE_STATUSES,
    RecurringPattern,
    can_reduce,
    can_stop,
)

HORIZON_DAYS = 90


@dataclass
class CashEvent:
    date: pd.Timestamp
    amount_signed: float  # positive = credit, negative = debit
    category: str
    event_id: str | None
    source: str  # "actual" | "recurring_projection" | "candidate_payment" | "salary"


@dataclass
class SimulationResult:
    safe: bool
    min_balance: float
    min_balance_date: pd.Timestamp | None
    daily_balances: list[tuple[pd.Timestamp, float]] = field(default_factory=list)


def _project_recurring(
    patterns: list[RecurringPattern],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    existing_window_events: pd.DataFrame,
    spending_changes: dict[str, tuple[str, float | None]] | None = None,
    dedup_window_days: int = 5,
) -> list[CashEvent]:
    """
    spending_changes: {event_id: ("stop", None) or ("reduce_to", new_amount)}
    Applied by matching the pattern's category — if ANY sample event id of a
    pattern is targeted by a spending change, that change is applied to every
    projected occurrence of that pattern within the window.
    """
    spending_changes = spending_changes or {}
    out: list[CashEvent] = []

    for pat in patterns:
        # does this pattern have a spending change targeting it?
        change = None
        for eid in pat.sample_event_ids:
            if eid in spending_changes:
                change = spending_changes[eid]
                break

        # existing actual events of the same category/direction inside window
        same_cat = existing_window_events[
            (existing_window_events["category"] == pat.category)
            & (existing_window_events["direction"] == pat.direction)
        ]
        existing_dates = list(same_cat["event_date"])

        next_date = pat.last_date + pd.Timedelta(days=pat.interval_days)
        while next_date <= end_date:
            if next_date >= start_date:
                # skip if an actual event already covers this occurrence
                is_dup = any(
                    abs((next_date - d).days) <= dedup_window_days
                    for d in existing_dates
                )
                if not is_dup:
                    amount = pat.typical_amount
                    if change is not None:
                        kind, new_amount = change
                        if kind == "stop":
                            amount = 0.0
                        elif kind == "reduce_to":
                            amount = new_amount
                    signed = amount if pat.direction == "credit" else -amount
                    out.append(
                        CashEvent(
                            date=next_date,
                            amount_signed=signed,
                            category=pat.category,
                            event_id=None,
                            source="recurring_projection",
                        )
                    )
            next_date += pd.Timedelta(days=pat.interval_days)

    return out


def _actual_window_events(
    events: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> list[CashEvent]:
    """
    Confirmed one-off events already in the data that fall in the window.
    Only scheduled/pending/settled-in-future rows count (per the safety
    check: ignore pending credits, failed/cancelled, duplicates,
    unrealized investments).
    """
    out = []
    window = events[
        (events["event_date"] >= start_date) & (events["event_date"] <= end_date)
    ]
    for _, row in window.iterrows():
        if row["status"] not in CASH_SAFE_STATUSES:
            continue
        if row["direction"] == "non_cash":
            continue
        # Ignore pending CREDITS specifically (bonuses/commissions/refunds/
        # investment gains not yet settled) but keep pending DEBITS
        # (reserve them) per AGENTS.md 6.3.
        if row["status"] == "pending" and row["direction"] == "credit":
            continue
        if pd.isna(row["amount_home"]):
            continue  # unresolved blank amount — caller should have filled this
        signed = row["amount_home"] if row["direction"] == "credit" else -row["amount_home"]
        out.append(
            CashEvent(
                date=row["event_date"],
                amount_signed=signed,
                category=row["category"],
                event_id=row["event_id"],
                source="actual",
            )
        )
    return out


def simulate(
    start_balance: float,
    start_date: pd.Timestamp,
    events_df: pd.DataFrame,
    patterns: list[RecurringPattern],
    minimum_balance_to_keep: float,
    horizon_days: int = HORIZON_DAYS,
    extra_cash_events: list[CashEvent] | None = None,
    spending_changes: dict[str, tuple[str, float | None]] | None = None,
) -> SimulationResult:
    """
    Runs the 90-day forecast starting at start_date with start_balance,
    applying: actual future events in the window, projected recurring
    events, and any extra_cash_events (e.g. the candidate payment(s) under
    test). Returns whether the balance ever dips below
    minimum_balance_to_keep.
    """
    end_date = start_date + pd.Timedelta(days=horizon_days)

    all_events = list(_actual_window_events(events_df, start_date, end_date))
    all_events += _project_recurring(
        patterns,
        start_date,
        end_date,
        existing_window_events=events_df[
            (events_df["event_date"] >= start_date) & (events_df["event_date"] <= end_date)
        ],
        spending_changes=spending_changes,
    )
    if extra_cash_events:
        all_events += extra_cash_events

    all_events.sort(key=lambda e: e.date)

    balance = start_balance
    daily_balances = [(start_date, balance)]
    min_balance = balance
    min_balance_date = start_date

    for ev in all_events:
        balance += ev.amount_signed
        daily_balances.append((ev.date, balance))
        if balance < min_balance:
            min_balance = balance
            min_balance_date = ev.date

    safe = min_balance >= minimum_balance_to_keep
    return SimulationResult(
        safe=safe,
        min_balance=min_balance,
        min_balance_date=min_balance_date,
        daily_balances=daily_balances,
    )
