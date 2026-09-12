"""
Generates decision_explanation as natural-language sentences, not a
telegraphic fact-dump. No LLM call here — the phrasing patterns below
were reverse-engineered directly from dataset/sample_requests.csv (25
ground-truth examples), grouped by (affordability_status,
recommended_payment_method). Doing this with string templates instead of
an LLM call keeps it free and, more importantly, GUARANTEED consistent
with the actual computed numbers — an LLM asked to "explain" the decision
could still hallucinate a figure that doesn't match amount_safe_to_pay.

Observed patterns (see each builder function for the matching example):
  affordable_now, full_payment              -> _explain_full_payment_now
  affordable_with_plan, full_payment+changes -> _explain_full_payment_with_changes
  affordable_with_plan, installments         -> _explain_installments
  affordable_with_plan, partial_payment      -> _explain_partial_payment
  affordable_later, wait                     -> _explain_wait
  not_affordable (amount_safe_to_pay == 0)   -> _explain_not_affordable_none_safe
  not_affordable (amount_safe_to_pay > 0)    -> _explain_not_affordable_partial_only
"""
from __future__ import annotations

import pandas as pd


def format_money(amount: float, currency: str) -> str:
    if abs(amount - round(amount)) < 0.005:
        return f"{currency} {round(amount):,}"
    return f"{currency} {amount:,.2f}"


def format_date(ts: pd.Timestamp) -> str:
    return f"{ts.day} {ts.strftime('%B')} {ts.year}"


def _event_description(ev: pd.DataFrame, event_id: str) -> str:
    match = ev[ev["event_id"] == event_id]
    if match.empty:
        return "the recurring expense"
    desc = match.iloc[0].get("description")
    return desc if isinstance(desc, str) and desc else "the recurring expense"


def _spending_change_phrases(ev: pd.DataFrame, spending_changes_str: str, currency: str) -> list[str]:
    """"stop:event_476|reduce_to:event_1816:23.5" -> ["stop the family
    streaming plan", "reduce the streaming subscription to USD 23.50"]"""
    if spending_changes_str in ("", "none"):
        return []
    phrases = []
    for part in spending_changes_str.split("|"):
        pieces = part.split(":")
        kind = pieces[0]
        event_id = pieces[1]
        desc = _event_description(ev, event_id).lower()
        if kind == "stop":
            phrases.append(f"stop the {desc}")
        elif kind == "reduce_to":
            new_amount = float(pieces[2])
            phrases.append(f"reduce the {desc} to {format_money(new_amount, currency)}")
    return phrases


def _join_and(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _explain_full_payment_now(currency: str, amount: float, min_balance: float) -> str:
    # matches: "Pay ZAR 25,256 today. This leaves at least ZAR 18,000
    # available over the next 90 days." (request_01)
    return (f"Pay {format_money(amount, currency)} today. This leaves at least "
            f"{format_money(min_balance, currency)} available over the next 90 days.")


def _explain_full_payment_with_changes(currency: str, amount: float, min_balance: float,
                                        change_phrases: list[str]) -> str:
    # matches: "Stop the family streaming plan, then pay EUR 620.40 today.
    # This leaves at least EUR 800 available." (request_06)
    lead = _join_and([p.capitalize() if i == 0 else p for i, p in enumerate(change_phrases)])
    return (f"{lead}, then pay {format_money(amount, currency)} today. "
            f"This leaves at least {format_money(min_balance, currency)} available.")


def _explain_installments(currency: str, n_payments: int, per_payment: float,
                           start_date: pd.Timestamp, min_balance: float) -> str:
    # matches: "Use 3 installments of IDR 15,952,906.67, starting 8 August
    # 2025. This leaves at least IDR 29,158,400 available." (request_02)
    return (f"Use {n_payments} installments of {format_money(per_payment, currency)}, "
            f"starting {format_date(start_date)}. This leaves at least "
            f"{format_money(min_balance, currency)} available.")


def _explain_partial_payment(currency: str, first_amount: float, second_amount: float,
                              second_date: pd.Timestamp, min_balance: float) -> str:
    # matches: "Pay INR 28,820 today and the remaining INR 10,840 on 15
    # September 2024. This completes the full request and keeps the INR
    # 92,800 minimum protected." (request_19)
    return (f"Pay {format_money(first_amount, currency)} today and the remaining "
            f"{format_money(second_amount, currency)} on {format_date(second_date)}. "
            f"This completes the full request and keeps the "
            f"{format_money(min_balance, currency)} minimum protected.")


def _explain_wait(currency: str, amount: float, pay_date: pd.Timestamp, min_balance: float) -> str:
    # matches: "Pay IDR 5,491,000 in full on 15 November 2019. Paying
    # earlier would take the balance below the IDR 2,668,700 minimum."
    # (request_03)
    return (f"Pay {format_money(amount, currency)} in full on {format_date(pay_date)}. "
            f"Paying earlier would take the balance below the "
            f"{format_money(min_balance, currency)} minimum.")


def _explain_not_affordable_none_safe(currency: str, deadline: pd.Timestamp, min_balance: float) -> str:
    # matches: "Do not make this payment by 12 January 2026. None of the
    # available options keeps the ZAR 13,100 minimum protected."
    # (request_05)
    return (f"Do not make this payment by {format_date(deadline)}. None of the "
            f"available options keeps the {format_money(min_balance, currency)} minimum protected.")


def _explain_not_affordable_partial_only(currency: str, requested_amount: float, safe_amount: float) -> str:
    # matches: "Do not proceed with the EUR 5,414.20 request. Although EUR
    # 597.74 is available today, the full amount cannot be completed
    # safely within 90 days." (request_14)
    return (f"Do not proceed with the {format_money(requested_amount, currency)} request. "
            f"Although {format_money(safe_amount, currency)} is available today, the full "
            f"amount cannot be completed safely within 90 days.")


def build_decision_explanation(*, status: str, method: str, currency: str,
                                ev: pd.DataFrame, prof: dict, req_row,
                                safe_amount: float, winner, injection_note: str = "") -> str:
    min_balance = prof["minimum_balance_to_keep"]
    requested_amount = req_row["requested_amount"]
    deadline = req_row["desired_completion_date"]

    if winner is None:
        if safe_amount > 0:
            text = _explain_not_affordable_partial_only(currency, requested_amount, safe_amount)
        else:
            text = _explain_not_affordable_none_safe(currency, deadline, min_balance)
        return text + injection_note

    if method == "wait":
        pay_date, amount = winner.payments[0]
        text = _explain_wait(currency, amount, pay_date, min_balance)

    elif method == "installments":
        pay_date, per_payment = winner.payments[0]
        text = _explain_installments(currency, len(winner.payments), per_payment, pay_date, min_balance)

    elif method == "partial_payment":
        (d1, a1), (d2, a2) = winner.payments[0], winner.payments[1]
        text = _explain_partial_payment(currency, a1, a2, d2, min_balance)

    elif method == "full_payment" and winner.requires_spending_change:
        _, amount = winner.payments[0]
        phrases = _spending_change_phrases(ev, winner.spending_changes_str, currency)
        text = _explain_full_payment_with_changes(currency, amount, min_balance, phrases)

    else:  # full_payment, no changes
        _, amount = winner.payments[0]
        text = _explain_full_payment_now(currency, amount, min_balance)

    return text + injection_note
