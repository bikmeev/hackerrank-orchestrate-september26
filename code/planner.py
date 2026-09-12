"""
Builds candidate payment plans (full_payment / installments / partial_payment
/ wait), checks each against the 90-day safety check, filters by the user's
payment_methods_user_will_consider + max_installment_months, and ranks the
safe+eligible candidates using the 6-rule order from the problem statement:

  1. Complete the full request by desired_completion_date.
  2. Require no spending changes.
  3. Minimize the total amount paid.
  4. Start payment earlier.
  5. Use fewer payments.
  6. Use the lowest payment_option_id as the final tie-breaker.

This module does NOT touch messages/images/spending-changes yet — those
plug in as additional candidates / corrections upstream. For now
"require no spending changes" is trivially true for every candidate here
since none of them include spending changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from forecast import CashEvent, simulate


@dataclass
class Candidate:
    method: str  # full_payment | installments | partial_payment | wait
    payments: list[tuple[pd.Timestamp, float]]  # chronological (date, amount)
    payment_option_id: str | None  # for installments, used as final tie-breaker
    completes_by_deadline: bool
    requires_spending_change: bool = False
    spending_changes_str: str = "none"
    safe: bool = False
    min_balance: float = 0.0

    @property
    def total_paid(self) -> float:
        return sum(a for _, a in self.payments)

    @property
    def start_date(self) -> pd.Timestamp:
        return self.payments[0][0] if self.payments else pd.Timestamp.max

    @property
    def num_payments(self) -> int:
        return len(self.payments)

    def format_plan(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(f"{d.date()}:{round(a, 2)}" for d, a in self.payments)


def _installment_payments(option_row: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    n = int(option_row["number_of_payments"])
    first = option_row["first_payment_date"]
    freq = option_row["payment_frequency_days"]
    amount = float(option_row["payment_amount"])
    payments = []
    for i in range(n):
        date = first if i == 0 else first + pd.Timedelta(days=int(freq) * i)
        payments.append((date, amount))
    return payments


def _check_safety(prof, ev, patterns, req_date, payments: list[tuple[pd.Timestamp, float]],
                   spending_changes: dict | None = None) -> tuple[bool, float]:
    extra = [
        CashEvent(date=d, amount_signed=-a, category="plan_payment",
                  event_id=None, source="candidate_payment")
        for d, a in payments
    ]
    res = simulate(
        prof["current_available_balance"], req_date, ev, patterns,
        prof["minimum_balance_to_keep"], extra_cash_events=extra,
        spending_changes=spending_changes,
    )
    return res.safe, res.min_balance


def _pattern_change_option(p) -> tuple[str, float | None, float] | None:
    """Returns (kind, new_amount, savings) for the strongest available cut
    on this pattern, or None if it isn't flexible / has no usable floor."""
    if p.flexibility == "stoppable":
        return "stop", None, p.typical_amount
    if p.flexibility == "reducible":
        if p.minimum_allowed_amount is None:
            return None  # no known floor — don't guess one
        return "reduce_to", p.minimum_allowed_amount, max(0.0, p.typical_amount - p.minimum_allowed_amount)
    if p.flexibility == "reducible_or_stoppable":
        return "stop", None, p.typical_amount  # stop dominates reduce_to on savings
    return None  # "fixed" — not changeable


def find_minimal_spending_changes(prof, ev, patterns, req_date, target_date, amount,
                                   max_changes: int = 3):
    """
    Greedily tries the biggest-saving flexible patterns first, adding cuts
    (up to max_changes) until paying `amount` on `target_date` becomes
    safe. Returns (spending_changes_dict, formatted_string) on success, or
    None if it can't be made safe within max_changes cuts.

    spending_changes_dict is keyed by event_id — the pattern's most recent
    real occurrence — matching forecast.py's expected format AND the
    output field's expected reference (verified against
    dataset/sample_requests.csv: spending_changes_needed points at the
    most recent settled event_id of the affected category, e.g.
    "stop:event_476").
    """
    options = []
    for p in patterns:
        opt = _pattern_change_option(p)
        if opt is None:
            continue
        kind, new_amount, savings = opt
        if savings <= 0:
            continue
        options.append((savings, p, kind, new_amount))
    options.sort(key=lambda t: -t[0])  # biggest savings first

    changes: dict[str, tuple[str, float | None]] = {}
    formatted_parts = []
    for savings, p, kind, new_amount in options[:max_changes]:
        event_id = p.sample_event_ids[-1]
        changes[event_id] = (kind, new_amount)
        formatted_parts.append(
            f"stop:{event_id}" if kind == "stop" else f"reduce_to:{event_id}:{new_amount}"
        )
        safe, _ = _check_safety(prof, ev, patterns, req_date, [(target_date, amount)],
                                 spending_changes=changes)
        if safe:
            return changes, "|".join(formatted_parts)
    return None


def _max_installment_days(prof) -> float | None:
    months = prof.get("max_installment_months")
    if pd.isna(months):
        return None
    return float(months) * 30.0  # approximate month length, consistent with ~30d intervals in the data


def build_candidates(prof, ev, patterns, req_row, payment_options: pd.DataFrame,
                      amount_safe_to_pay: float,
                      earliest_full_date: pd.Timestamp | None) -> list[Candidate]:
    req_date = req_row["request_date"]
    desired_date = req_row["desired_completion_date"]
    requested_amount = req_row["requested_amount"]
    accepted_methods = str(prof["payment_methods_user_will_consider"]).split("|")

    candidates: list[Candidate] = []

    # --- full_payment today ---
    full_opt = payment_options[payment_options["payment_method"] == "full_payment"]
    if not full_opt.empty and "full_payment" in accepted_methods:
        row = full_opt.iloc[0]
        payments = [(row["first_payment_date"], float(row["payment_amount"]))]
        safe, min_bal = _check_safety(prof, ev, patterns, req_date, payments)
        if safe:
            candidates.append(Candidate(
                method="full_payment",
                payments=payments,
                payment_option_id=row["payment_option_id"],
                completes_by_deadline=payments[0][0] <= desired_date,
                safe=True,
                min_balance=min_bal,
            ))
        else:
            # Not safe as-is — see if cutting up to 3 flexible expenses
            # makes paying the FULL amount today safe. If so, this is an
            # "affordable_with_plan" full_payment, not affordable_now.
            found = find_minimal_spending_changes(prof, ev, patterns, req_date,
                                                    row["first_payment_date"], float(row["payment_amount"]))
            if found is not None:
                changes, formatted = found
                safe2, min_bal2 = _check_safety(prof, ev, patterns, req_date, payments,
                                                  spending_changes=changes)
                candidates.append(Candidate(
                    method="full_payment",
                    payments=payments,
                    payment_option_id=row["payment_option_id"],
                    completes_by_deadline=payments[0][0] <= desired_date,
                    requires_spending_change=True,
                    spending_changes_str=formatted,
                    safe=safe2,
                    min_balance=min_bal2,
                ))

    # --- wait: full payment later, at the earliest safe date ---
    # Only a distinct candidate if it's actually LATER than today — if the
    # earliest safe date is req_date itself, that's identical to the
    # full_payment candidate above and would create a spurious tie.
    if (earliest_full_date is not None and earliest_full_date > req_date
            and "full_payment" in accepted_methods):
        payments = [(earliest_full_date, requested_amount)]
        safe, min_bal = _check_safety(prof, ev, patterns, req_date, payments)
        candidates.append(Candidate(
            method="wait",
            payments=payments,
            payment_option_id=None,
            completes_by_deadline=earliest_full_date <= desired_date,
            safe=safe,
            min_balance=min_bal,
        ))

    # --- installments: must match a supplied option exactly ---
    if "installments" in accepted_methods:
        max_days = _max_installment_days(prof)
        inst_opts = payment_options[payment_options["payment_method"] == "installments"]
        for _, row in inst_opts.iterrows():
            duration_days = float(row["payment_frequency_days"]) * (int(row["number_of_payments"]) - 1)
            if max_days is not None and duration_days > max_days:
                continue  # user's max_installment_months rejects this option
            payments = _installment_payments(row)
            last_date = payments[-1][0]
            safe, min_bal = _check_safety(prof, ev, patterns, req_date, payments)
            candidates.append(Candidate(
                method="installments",
                payments=payments,
                payment_option_id=row["payment_option_id"],
                completes_by_deadline=last_date <= desired_date,
                safe=safe,
                min_balance=min_bal,
            ))

    # --- partial_payment: exactly 2 payments, only if the request allows it ---
    allows_partial = bool(req_row["allows_partial_payment"])
    if (allows_partial and "partial_payment" in accepted_methods
            and 0 < amount_safe_to_pay < requested_amount
            and earliest_full_date is not None
            and earliest_full_date <= desired_date):
        remaining = requested_amount - amount_safe_to_pay
        payments = [(req_date, amount_safe_to_pay), (earliest_full_date, remaining)]
        safe, min_bal = _check_safety(prof, ev, patterns, req_date, payments)
        candidates.append(Candidate(
            method="partial_payment",
            payments=payments,
            payment_option_id=None,
            completes_by_deadline=earliest_full_date <= desired_date,
            safe=safe,
            min_balance=min_bal,
        ))

    return candidates


def rank_candidates(candidates: list[Candidate]) -> Candidate | None:
    """
    Applies the 6-rule ranking to the SAFE candidates only. Returns the
    winning candidate, or None if no safe+eligible candidate exists
    (caller should fall back to not_recommended / not_affordable).
    """
    safe = [c for c in candidates if c.safe]
    if not safe:
        return None

    def sort_key(c: Candidate):
        return (
            0 if c.completes_by_deadline else 1,      # rule 1: deadline (lower=better)
            0 if not c.requires_spending_change else 1,  # rule 2
            c.total_paid,                               # rule 3: minimize total paid
            c.start_date,                                # rule 4: start earlier
            c.num_payments,                              # rule 5: fewer payments
            c.payment_option_id or "",                   # rule 6: lowest option id
        )

    safe.sort(key=sort_key)
    return safe[0]
