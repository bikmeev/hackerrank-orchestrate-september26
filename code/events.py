"""
Turns raw financial_events.csv rows into a normalized, home-currency view,
and detects recurring expense/income patterns so we can project them
forward for the 90-day forecast.

Key data facts confirmed from the dataset:
- status in {settled, pending, scheduled, cancelled, failed, unrealized}
- flexibility in {fixed, reducible, reducible_or_stoppable, stoppable}
- direction in {debit, credit, non_cash}
- event_type in {expense, income, debt_payment, subscription,
  investment_purchase, investment_sale, investment_valuation, refund}
- 16 rows have a blank `amount` -> must be resolved from a linked image
  via images.csv (related_event_id == event_id). Never treat as 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from currency import CurrencyConverter

# Per AGENTS.md 6.4 / problem_statement "90-Day Safety Check":
# ignore pending credits, failed/cancelled, duplicates, unrealized investments.
CASH_SAFE_STATUSES = {"settled", "scheduled", "pending"}
EXCLUDED_FROM_FORECAST_STATUSES = {"cancelled", "failed", "unrealized"}

FLEXIBLE_KINDS = {"reducible", "reducible_or_stoppable", "stoppable"}


def is_flexible(flexibility: str) -> bool:
    return flexibility in FLEXIBLE_KINDS


def can_stop(flexibility: str) -> bool:
    return flexibility in {"stoppable", "reducible_or_stoppable"}


def can_reduce(flexibility: str) -> bool:
    return flexibility in {"reducible", "reducible_or_stoppable"}


def resolve_blank_amounts(events: pd.DataFrame, images: pd.DataFrame,
                           image_amount_lookup: dict) -> pd.DataFrame:
    """
    For events with a blank amount, look up dataset/media/images/<image_id>.png
    via images.csv (related_event_id == event_id) and fill in the amount.

    `image_amount_lookup` is a dict {image_id: amount} that the caller has
    already extracted (e.g. via a vision-capable LLM call) — this function
    stays purely mechanical/deterministic and does not do extraction itself.
    """
    events = events.copy()
    blank_mask = events["amount"].isna()
    for idx in events[blank_mask].index:
        event_id = events.at[idx, "event_id"]
        match = images[images["related_event_id"] == event_id]
        if match.empty:
            continue  # leave as NaN; caller should decide how to handle
        image_id = match.iloc[0]["image_id"]
        if image_id in image_amount_lookup:
            events.at[idx, "amount"] = image_amount_lookup[image_id]
    return events


def normalize_to_home_currency(events: pd.DataFrame, home_currency: str,
                                converter: CurrencyConverter) -> pd.DataFrame:
    """Adds an `amount_home` column: amount converted to the user's home
    currency using the exact settlement_date rate."""
    events = events.copy()

    def _convert(row):
        if pd.isna(row["amount"]):
            return row["amount"]
        if row["currency"] == home_currency:
            return row["amount"]
        return converter.convert(
            row["amount"], row["currency"], home_currency, row["settlement_date"]
        )

    events["amount_home"] = events.apply(_convert, axis=1)
    return events


def apply_amendments(events: pd.DataFrame, amendments: list[dict]) -> tuple[pd.DataFrame, set[str]]:
    """
    Applies LLM-extracted amendments (see llm_extract.EXTRACTION_SYSTEM_PROMPT
    schema) to a user's raw event rows, BEFORE currency normalization and
    recurrence detection.

    Returns (modified_events, suppressed_categories):
      - modified_events: the events DataFrame with amendments applied
      - suppressed_categories: category names that should NOT be projected
        forward by detect_recurring_patterns (e.g. "salary" after a message
        says the job/contract ended) — the caller is responsible for
        filtering detect_recurring_patterns() output against this set,
        since suppression is a forecasting decision, not a data mutation.

    - target "event:<id>": mutates that row in place (cancel -> status
      change; amend_amount/amend_date -> overwrite amount/currency or
      event_date/settlement_date). This is also how blank-amount events
      get resolved from an image.
    - target "unknown": a genuine one-off fact (confirmed invoice, bonus,
      refund) with no matching event/category. If amount + date +
      direction are all present, injects a single one-time synthetic
      event (not a recurring pattern — it's a plain actual event, so the
      90-day forecast picks it up directly without going through
      detect_recurring_patterns). Skipped if any of the three is missing,
      rather than guessing a sign or a date.
    - "confirm"/"new_info" on an event target: no structural change
      (confirmations are supposed to be no-ops).
    """
    events = events.copy()
    synthetic_rows = []
    suppressed_categories: set[str] = set()

    def _flexibility_default(category: str) -> str:
        precedent = events[events["category"] == category]
        if not precedent.empty:
            return precedent.iloc[-1]["flexibility"]
        return "fixed"  # conservative default for a category we've never seen

    for amendment in amendments:
        target = amendment.get("target", "unknown")
        action = amendment.get("action", "new_info")

        if target.startswith("event:"):
            event_id = target.split(":", 1)[1]
            mask = events["event_id"] == event_id
            if not mask.any():
                continue
            if action == "cancel":
                events.loc[mask, "status"] = "cancelled"
            elif action in ("amend_amount",):
                if amendment.get("amount") is not None:
                    events.loc[mask, "amount"] = amendment["amount"]
                    if amendment.get("currency"):
                        events.loc[mask, "currency"] = amendment["currency"]
            elif action in ("amend_date", "delay"):
                if amendment.get("date"):
                    new_date = pd.Timestamp(amendment["date"])
                    events.loc[mask, "event_date"] = new_date
                    events.loc[mask, "settlement_date"] = new_date
            # "confirm" / "new_info" -> no-op by design

        elif target.startswith("category:"):
            category = target.split(":", 1)[1]
            if action == "cancel":
                suppressed_categories.add(category)
                continue
            if action not in ("amend_amount", "amend_date", "delay", "new_info"):
                continue
            precedent = events[events["category"] == category]
            if precedent.empty or amendment.get("amount") is None or amendment.get("date") is None:
                continue
            last = precedent.sort_values("event_date").iloc[-1]
            new_date = pd.Timestamp(amendment["date"])
            synthetic_rows.append({
                "event_id": f"llm_synth_{category}_{new_date.date()}",
                "user_id": last["user_id"],
                "event_type": last["event_type"],
                "description": f"LLM-extracted amendment: {amendment.get('reasoning', '')}",
                "category": category,
                "direction": last["direction"],
                "amount": amendment["amount"],
                "currency": amendment.get("currency") or last["currency"],
                "event_date": new_date,
                "settlement_date": new_date,
                "status": "scheduled",
                "linked_event_id": None,
                "flexibility": last["flexibility"],
                "minimum_allowed_amount": last.get("minimum_allowed_amount"),
            })

        elif target == "unknown":
            amount = amendment.get("amount")
            date = amendment.get("date")
            direction = amendment.get("direction")
            if amount is None or not date or direction not in ("credit", "debit"):
                continue  # not enough to safely add — skip rather than guess
            new_date = pd.Timestamp(date)
            category = "one_off_income" if direction == "credit" else "one_off_expense"
            user_id = events["user_id"].iloc[0] if not events.empty else None
            synthetic_rows.append({
                "event_id": f"llm_synth_unknown_{new_date.date()}",
                "user_id": user_id,
                "event_type": "income" if direction == "credit" else "expense",
                "description": f"LLM-extracted one-off: {amendment.get('reasoning', '')}",
                "category": category,
                "direction": direction,
                "amount": amount,
                "currency": amendment.get("currency"),
                "event_date": new_date,
                "settlement_date": new_date,
                "status": "scheduled",
                "linked_event_id": None,
                "flexibility": "fixed",  # one-off facts aren't adjustable spending
                "minimum_allowed_amount": None,
            })

    if synthetic_rows:
        events = pd.concat([events, pd.DataFrame(synthetic_rows)], ignore_index=True)
    return events, suppressed_categories


@dataclass
class RecurringPattern:
    category: str
    direction: str
    typical_amount: float
    interval_days: int
    last_date: pd.Timestamp
    flexibility: str
    sample_event_ids: list
    minimum_allowed_amount: float | None = None


def _longest_regular_chain(dates: list[pd.Timestamp], interval_tolerance_days: int):
    """
    Finds the longest "chain" of dates where each consecutive pair is
    spaced ~interval_days apart (a consistent recurring interval),
    tolerating one-off outliers (bonuses, corrections, missed months)
    that don't fit the grid. Returns (chain_indices, interval_days) for
    the best chain found, or (None, None) if nothing qualifies.

    Approach: try every pair of dates as a candidate (start, interval),
    then greedily extend forward picking the next date that best matches
    "previous chain date + interval" within tolerance. This is O(n^2)
    per category, which is fine at this dataset's scale (a few dozen
    events per user per category at most).
    """
    n = len(dates)
    best_chain = None
    best_interval = None

    for i in range(n):
        for j in range(i + 1, n):
            interval = (dates[j] - dates[i]).days
            if interval <= 0:
                continue
            chain = [i]
            cursor = dates[i]
            k = j
            while k < n:
                # find the next date close to cursor + interval, scanning forward
                target = cursor + pd.Timedelta(days=interval)
                best_k = None
                best_diff = interval_tolerance_days + 1
                for m in range(k, n):
                    diff = abs((dates[m] - target).days)
                    if diff <= interval_tolerance_days and diff < best_diff:
                        best_k = m
                        best_diff = diff
                if best_k is None:
                    break
                chain.append(best_k)
                cursor = dates[best_k]
                k = best_k + 1
            if best_chain is None or len(chain) > len(best_chain):
                best_chain = chain
                best_interval = interval

    return best_chain, best_interval


def _trailing_consistent_amount(chain_rows: pd.DataFrame, amount_tolerance_pct: float,
                                 trailing_min: int = 2, strong_run: int = 3):
    """
    Walks backward from the most recent occurrence, extending the trailing
    "current regime" as long as each additional (older) amount stays
    within tolerance of the running median.

    If the trailing run reaches `strong_run` (default 3) occurrences on
    its own, that's accepted outright — 3+ agreeing values is solid
    evidence regardless of what came before.

    If extension stops earlier (only `trailing_min`, i.e. 2, agree), that
    is ONLY accepted as a genuine rate change if the excluded OLDER
    remainder is itself internally consistent (forms a coherent "old
    regime" of its own) — e.g. salary at 1441 x3 then a step down to
    1037.5 x2: the old segment is perfectly stable, so the step is real.
    Without that check, a merely noisy category (e.g. dining, which
    fluctuates ~915-1244 with no real step) would occasionally have two
    adjacent values land within tolerance of each other by chance and get
    mistaken for a "recent rate change" — the older-consistency check
    tells those two situations apart.
    """
    amounts = chain_rows["amount_home"].tolist()
    ids = chain_rows["event_id"].tolist()
    n = len(amounts)
    if n < trailing_min:
        return None

    included = list(range(n - trailing_min, n))
    median = pd.Series([amounts[i] for i in included]).median()
    if any(abs(amounts[i] - median) > amount_tolerance_pct * abs(median) for i in included):
        return None  # even the most recent occurrences disagree — no stable pattern at all

    for i in range(n - trailing_min - 1, -1, -1):
        candidate_included = [i] + included
        candidate_median = pd.Series([amounts[j] for j in candidate_included]).median()
        if all(abs(amounts[j] - candidate_median) <= amount_tolerance_pct * abs(candidate_median)
               for j in candidate_included):
            included = candidate_included
            median = candidate_median
        else:
            break

    if len(included) >= strong_run:
        return float(median), [ids[i] for i in included]

    # short trailing run — only trust it if the older remainder is itself
    # a coherent, internally-consistent regime (a real step), not just noise
    excluded = [i for i in range(n) if i not in included]
    if len(excluded) < 2:
        return None
    excluded_amounts = [amounts[i] for i in excluded]
    excluded_median = pd.Series(excluded_amounts).median()
    excluded_consistent = all(
        abs(a - excluded_median) <= amount_tolerance_pct * abs(excluded_median)
        for a in excluded_amounts
    )
    if not excluded_consistent:
        return None

    return float(median), [ids[i] for i in included]


def detect_recurring_patterns(events: pd.DataFrame,
                               min_occurrences: int = 3,
                               interval_tolerance_days: int = 6,
                               amount_tolerance_pct: float = 0.15) -> list[RecurringPattern]:
    """
    Heuristic recurrence detection for ONE user's events.

    Groups settled/scheduled debit or credit events by (category, direction)
    and finds the longest chain of roughly-equally-spaced occurrences
    (see _longest_regular_chain) — this tolerates outliers like a one-off
    bonus or a blank-amount row sharing the same category, which would
    otherwise break a naive "every event must be consistent" check.

    The chain's INTERVAL must hold across the full chain (>= min_occurrences
    occurrences), but its AMOUNT only needs to be consistent across the
    most recent occurrences (see _trailing_consistent_amount) — so a
    permanent rate change (raise, pay cut, rent increase) doesn't reject
    the whole pattern just because older occurrences no longer match.

    Salary IS included here: the dataset has no separate "next confirmed
    salary" field, so the next salary occurrence is itself detected as a
    recurring credit pattern (typically monthly, ~30-31 day interval).
    Message/image evidence (e.g. a payroll letter announcing a raise or a
    delay) should later override the projected amount/date for salary —
    that correction happens upstream of this function, by adjusting the
    user's event history before detection, not inside it.
    """
    patterns = []
    base = events[
        events["status"].isin(CASH_SAFE_STATUSES)
        & events["direction"].isin(["debit", "credit"])
        & events["amount_home"].notna()
    ].copy()
    base = base.sort_values("event_date")

    for (category, direction), group in base.groupby(["category", "direction"]):
        if len(group) < min_occurrences:
            continue
        group = group.reset_index(drop=True)
        dates = group["event_date"].tolist()

        chain, interval = _longest_regular_chain(dates, interval_tolerance_days)
        if chain is None or len(chain) < min_occurrences or interval is None or interval <= 0:
            continue

        chain_rows = group.iloc[chain].reset_index(drop=True)
        result = _trailing_consistent_amount(chain_rows, amount_tolerance_pct)
        if result is None:
            continue
        median_amount, used_event_ids = result

        last_row = chain_rows.iloc[-1]
        min_allowed = last_row.get("minimum_allowed_amount")
        patterns.append(
            RecurringPattern(
                category=category,
                direction=direction,
                typical_amount=median_amount,
                interval_days=int(round(interval)),
                last_date=last_row["event_date"],
                flexibility=last_row["flexibility"],
                sample_event_ids=used_event_ids,
                minimum_allowed_amount=float(min_allowed) if pd.notna(min_allowed) else None,
            )
        )
    return patterns
