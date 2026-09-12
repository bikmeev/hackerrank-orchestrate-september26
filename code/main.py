"""
Entry point: python3 code/main.py

Pipeline:
  1. load data
  2. per request: pull that user's message/image (if any) and run it
     through Claude Haiku 4.5 (llm_extract.py) to get structured
     amendments — confirmations, corrections, cancellations, blank-amount
     resolutions. Skipped gracefully (no crash) if ANTHROPIC_API_KEY isn't
     set, or if the user has neither a message nor an image.
  3. apply those amendments to the user's raw event history
  4. normalize events to home currency, detect recurring patterns
     (see events.py — this also detects salary as a recurring pattern,
     since the dataset has no separate "next confirmed salary" field)
  5. binary-search amount_safe_to_pay and the earliest safe date for the
     full amount (90-day safety check, forecast.py)
  6. build candidate plans (full_payment / wait / each installment option
     / partial_payment) and rank them per the 6-rule order (planner.py)
  7. write output.csv, and evaluation/usage_report.md summarizing the
     LLM calls made during this run

STILL NOT IMPLEMENTED (see README/AGENTS.md for full spec):
  - spending_changes_needed (stop:/reduce_to:) selection — currently
    always "none", so candidates never model an optional spending change
  - decision_explanation is factual but terse, not natural language
  - amendment currency conversion assumes the event's own settlement_date
    rate; doesn't yet handle a message stating a rate-sensitive amount on
    a date with no exchange_rates.csv row
"""
from __future__ import annotations

import argparse
import csv
import os

import pandas as pd

from currency import CurrencyConverter
from data_loader import load_data
from events import apply_amendments, detect_recurring_patterns, normalize_to_home_currency
from explanations import build_decision_explanation
from forecast import CashEvent, simulate
from llm_extract import LLMExtractor
from planner import build_candidates, rank_candidates

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def max_safe_amount(prof, ev, patterns, req_date, requested_amount, tol_steps=50):
    """Binary search the largest amount safe to pay today (90-day check)."""
    def is_safe(amt):
        candidate = [
            CashEvent(date=req_date, amount_signed=-amt, category="candidate",
                      event_id=None, source="candidate_payment")
        ]
        res = simulate(
            prof["current_available_balance"], req_date, ev, patterns,
            prof["minimum_balance_to_keep"], extra_cash_events=candidate,
        )
        return res.safe

    if requested_amount <= 0:
        return 0.0
    if not is_safe(0):
        return 0.0
    if is_safe(requested_amount):
        return requested_amount

    lo, hi = 0.0, requested_amount
    for _ in range(tol_steps):
        mid = (lo + hi) / 2
        if is_safe(mid):
            lo = mid
        else:
            hi = mid
    return lo


def earliest_full_payment_date(prof, ev, patterns, req_date, requested_amount,
                                horizon_days=90):
    """
    Scans forward day by day from req_date to find the first date where
    paying the FULL requested_amount on that date keeps the 90-day-from-THAT-date
    forecast safe. Returns None if never safe within the horizon.

    NOTE: this is O(horizon_days) simulate() calls per request — fine for
    a 250-row dataset, would want a smarter incremental approach at scale.
    """
    for offset in range(0, horizon_days + 1):
        candidate_date = req_date + pd.Timedelta(days=offset)
        candidate = [
            CashEvent(date=candidate_date, amount_signed=-requested_amount,
                      category="candidate", event_id=None, source="candidate_payment")
        ]
        res = simulate(
            prof["current_available_balance"], req_date, ev, patterns,
            prof["minimum_balance_to_keep"], extra_cash_events=candidate,
            horizon_days=horizon_days,
        )
        if res.safe:
            return candidate_date
    return None


def build_llm_context_events(ev: pd.DataFrame, related_event_ids: set[str]) -> list[dict]:
    """Keeps the LLM prompt small: only events the model plausibly needs to
    match against — blank amounts, salary/income history, and anything a
    message/image explicitly points at via related_event_id."""
    cols = ["event_id", "category", "direction", "amount", "currency",
            "event_date", "status"]
    mask = (
        ev["amount"].isna()
        | (ev["category"] == "salary")
        | ev["event_id"].isin(related_event_ids)
    )
    subset = ev[mask][cols].sort_values("event_date").tail(15)
    return subset.to_dict(orient="records")


def process_request(ctx, req_row, extractor: LLMExtractor) -> dict:
    uid = req_row["user_id"]
    prof = ctx.profile(uid)
    conv = ctx._converter  # attached in run()

    ev_raw = ctx.events_for_user(uid)

    # --- messages/images extraction & correction (before currency norm) ---
    msgs = ctx.messages_for(uid)
    imgs = ctx.images_for(uid)
    message_text = msgs.iloc[0]["message_text"] if not msgs.empty else None
    image_id = imgs.iloc[0]["image_id"] if not imgs.empty else None
    related_ids = set(msgs["related_event_id"].dropna()) | set(imgs["related_event_id"].dropna())

    extraction = extractor.extract(
        user_id=uid,
        message_text=message_text,
        image_id=image_id,
        context_events=build_llm_context_events(ev_raw, related_ids),
    )
    ev_raw, suppressed_categories = apply_amendments(ev_raw, extraction.get("amendments", []))

    ev = normalize_to_home_currency(ev_raw, prof["home_currency"], conv)
    patterns = detect_recurring_patterns(ev)
    patterns = [p for p in patterns if p.category not in suppressed_categories]

    req_date = req_row["request_date"]
    requested_amount = req_row["requested_amount"]

    safe_amount = max_safe_amount(prof, ev, patterns, req_date, requested_amount)
    earliest_dt = earliest_full_payment_date(prof, ev, patterns, req_date, requested_amount)

    payment_options = ctx.payment_options_for_request(req_row["request_id"])
    candidates = build_candidates(
        prof, ev, patterns, req_row, payment_options, safe_amount, earliest_dt
    )
    winner = rank_candidates(candidates)

    if winner is None:
        status = "not_affordable"
        method = "not_recommended"
        plan = "none"
        spending_changes = "none"
        earliest_out = earliest_dt if (earliest_dt is not None and earliest_dt <= req_row["desired_completion_date"]) else None
    else:
        method = winner.method
        plan = winner.format_plan()
        spending_changes = winner.spending_changes_str
        if method == "full_payment" and not winner.requires_spending_change:
            status = "affordable_now"
        elif method == "wait":
            status = "affordable_later"
        else:  # installments / partial_payment / full_payment-with-spending-change
            status = "affordable_with_plan"
        earliest_out = earliest_dt

    injection_note = ""
    if extraction.get("injection_attempt_detected"):
        injection_note = " [note: a linked message/image contained an embedded-instruction attempt; it was ignored, only genuine facts were used]"

    explanation = build_decision_explanation(
        status=status, method=method, currency=prof["home_currency"],
        ev=ev_raw, prof=prof, req_row=req_row, safe_amount=safe_amount,
        winner=winner, injection_note=injection_note,
    )

    return {
        "request_id": req_row["request_id"],
        "amount_safe_to_pay": round(safe_amount, 2),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan,
        "earliest_date_for_full_payment": earliest_out.date().isoformat() if earliest_out is not None else "",
        "spending_changes_needed": spending_changes,
        "decision_explanation": explanation,
    }


def run(dataset_dir: str, output_path: str, limit: int | None = None,
        cache_path: str = ".cache/llm_extractions.json",
        usage_report_path: str = "evaluation/usage_report.md"):
    ctx = load_data(dataset_dir)
    ctx._converter = CurrencyConverter(ctx.exchange_rates)  # small attach, avoids re-building per request

    media_dir = os.path.join(dataset_dir, "media", "images")
    extractor = LLMExtractor(cache_path=cache_path, media_dir=media_dir)

    requests = ctx.requests if limit is None else ctx.requests.head(limit)
    rows = []
    for _, req_row in requests.iterrows():
        try:
            rows.append(process_request(ctx, req_row, extractor))
        except Exception as e:
            # Never crash the whole run on one bad row — emit a safe
            # not_affordable fallback and keep going, so output.csv always
            # has one row per request_id as required.
            rows.append({
                "request_id": req_row["request_id"],
                "amount_safe_to_pay": 0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"[ERROR during processing: {e}]",
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")

    extractor.write_usage_report(usage_report_path, requests_processed=len(rows))
    print(f"Wrote usage report to {usage_report_path} "
          f"({extractor.usage.calls} LLM calls, "
          f"{extractor.usage.input_tokens + extractor.usage.output_tokens} tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--limit", type=int, default=None, help="process only first N requests (debug)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(here, "..", args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    output_path = os.path.join(here, "..", args.output) if not os.path.isabs(args.output) else args.output
    cache_path = os.path.join(here, "..", ".cache", "llm_extractions.json")
    usage_report_path = os.path.join(here, "..", "evaluation", "usage_report.md")

    run(dataset_dir, output_path, limit=args.limit,
        cache_path=cache_path, usage_report_path=usage_report_path)
