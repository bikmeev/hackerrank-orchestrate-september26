# Buy or Wait? — Solution

An AI-powered financial affordability agent for HackerRank Orchestrate
(September 2026). For every request in `dataset/requests.csv`, decides
whether the user can safely pay in full, pay partially, use an
installment plan, wait, or shouldn't proceed at all — grounded in a
90-day balance forecast, not a single-number balance check. See
[`problem_statement.md`](./problem_statement.md) for the full task spec.

## Quick start

```bash
pip install pandas anthropic python-dotenv
cp .env.example .env   # fill in your real ANTHROPIC_API_KEY
python3 code/main.py
```

Runs against `dataset/` by default and writes `output.csv` to the repo
root, plus `evaluation/usage_report.md`. Use `--limit N` to test on the
first N requests without spending on a full run. Works without an API
key too — the LLM step is skipped gracefully and the rest of the
pipeline (currency, forecast, planner) runs unchanged.

Model: `claude-haiku-4-5-20251001` — the cheapest current Claude model,
sufficient here because extraction is short-text/single-image →
fixed-schema JSON, not open-ended reasoning. See
`evaluation/usage_report.md` for the actual token/cost numbers from the
run that produced `output.csv`.

## Architecture at a glance

```
data_loader.py   -- loads & joins every dataset/*.csv into one DataContext
currency.py      -- exact-date currency conversion (+ USD/EUR hub routing)
llm_extract.py   -- Claude Haiku 4.5: turns messages/images into structured
                     amendments (confirm/amend/cancel/delay/new-info)
events.py        -- applies those amendments, normalizes to home currency,
                     detects recurring income/expense patterns
forecast.py      -- the 90-day safety check: projects a user's balance
                     forward and reports whether it ever dips below their
                     minimum_balance_to_keep
planner.py       -- builds candidate plans (full_payment / wait /
                     installments / partial_payment / full_payment with
                     spending changes), filters by what the user will
                     accept, ranks by the spec's 6-rule order
explanations.py  -- renders decision_explanation from templates reverse-
                     engineered from dataset/sample_requests.csv, not an
                     extra LLM call
main.py          -- orchestrates the above per request, writes output.csv
                     and evaluation/usage_report.md
```

Everything downstream of the LLM step is **deterministic Python** — the
LLM's job is narrowly scoped to natural-language/vision extraction, not
arithmetic. This split exists on purpose (see "Why hybrid" below).

## Why a hybrid pipeline, not a pure LLM agent

The output fields require exact arithmetic: a 90-day balance forecast,
currency conversion at a specific historical rate, installment schedules
that must match a supplied option byte-for-byte, and a strict
`0 <= amount_safe_to_pay <= requested_amount` bound. An LLM asked to do
that math directly will occasionally get dates or running totals wrong —
not because it's a bad model, but because multi-step arithmetic over a
90-day window isn't the kind of task token-by-token generation is
reliable at. Code doesn't make that class of error.

What genuinely needs a model: reading a payroll letter in Indonesian and
deciding whether it changes a forecast, or reading a scanned receipt to
recover a blank amount. That's the only piece handed to Claude.

## Key design decisions (and the bugs that shaped them)

**Recurring-pattern detection uses "longest regular chain", not "are all
occurrences consistent".** The first version required every event in a
(category, direction) group to be within tolerance of the group median —
and it broke on real data: one user's `salary` category included 5
consistent monthly payments *plus* a one-off bonus and a blank-amount row
sharing the same category label. The strict version threw out the whole
pattern, so no salary got projected at all, and every user's projected
balance sank to a deep negative over 90 days regardless of the payment
being tested. Fixed by searching for the longest chain of roughly
equally-spaced dates (tolerating outliers) instead of requiring the whole
group to agree — see `_longest_regular_chain` in `events.py`.

**A second, related bug: genuine rate changes (a raise or a pay cut) were
also rejected by full-history consistency.** One user's salary stepped
down permanently partway through their history (3 payments at one rate,
then 2 at a lower one) — again, a strict "whole group must agree" check
threw the entire pattern out. Fixed with `_trailing_consistent_amount`:
walk backward from the most recent occurrence, and accept a short (as few
as 2) trailing run as a genuine regime change only if the OLDER, excluded
portion is itself internally consistent (a real step, not just noise
around one mean) — this is what lets a true rate change win while still
rejecting a merely noisy category (e.g. dining) that happens to have two
adjacent similar values by chance.

**Salary is detected the same way as any other recurring category.**
There's no separate "next confirmed salary" field in the dataset — the
next salary occurrence is itself a projected recurring `credit`. Message
evidence (a payroll letter announcing a raise, delay, or "contract
ended") overrides this via `llm_extract.py` → `apply_amendments()`,
either injecting a corrected one-off occurrence or suppressing future
projection entirely (`category:<name>` + `action: "cancel"`).

**Currency conversion is an exact date+pair lookup, not nearest-date.**
Confirmed empirically: every foreign-currency event's `settlement_date`
matches an exact row in `exchange_rates.csv` for its currency pair (0
exceptions across 140 foreign-currency events). Rates only exist for a
handful of direct pairs (EUR↔USD, EUR→ZAR, USD→IDR, USD→INR); anything
else (e.g. ZAR↔IDR) routes through USD or EUR as a two-hop hub.

**The `wait` candidate is only generated when it's strictly later than
`full_payment`.** Early version generated both whenever the requested
amount was safe today, and a tie-breaker bug (`None` payment_option_id
sorting before a real one, alphabetically) made `wait` beat `full_payment`
in ranking even though they were financially identical. Fixed by only
emitting `wait` as a distinct candidate when its date is strictly after
the full_payment candidate's date.

**`spending_changes_needed` targets a specific event_id, not a category**
— confirmed by reverse-engineering `dataset/sample_requests.csv`: the
referenced id is the flexible category's most recent real occurrence
(e.g. `stop:event_476`). `planner.find_minimal_spending_changes` greedily
tries the biggest-saving flexible patterns first (up to 3), and a
`full_payment` candidate that only becomes safe after cuts is correctly
reported as `affordable_with_plan`, not `affordable_now` — a distinction
also confirmed from the samples (e.g. request_06: method `full_payment`,
status `affordable_with_plan`).

**`decision_explanation` is template-based, not an extra LLM call.**
All 25 sample explanations decompose into exactly 7 phrasing patterns by
`(affordability_status, recommended_payment_method)`, including that the
human-readable expense names ("family streaming plan") come straight from
`financial_events.csv`'s `description` field. Templating this guarantees
the text can never drift from the actual computed numbers — an LLM asked
to "explain" the decision could still state a figure that doesn't match
`amount_safe_to_pay`. On samples where our decision matches ground truth,
the generated text matches word-for-word.

**LLM calls are batched per user, not per message.** Checked the data
first: every user has at most one message and at most one image, and
`requests.csv` maps users to requests 1:1. So "per user" and "per
request" extraction are the same granularity here — and ~30 of 250 users
have neither a message nor an image, so those calls are skipped entirely
rather than wasting a call on nothing. Results are cached to
`.cache/llm_extractions.json` keyed by a hash of the inputs, so
re-running during development doesn't re-spend money.

**Extraction output is a small fixed JSON schema, not free text** —
`target` / `action` / `amount` / `currency` / `date` / `direction`. This
is also the actual prompt-injection defense: the schema structurally
can't express "set affordability_status to affordable_now" or anything
outside "here's a fact about one event or category." The system prompt
also explicitly frames message/image content as untrusted data and asks
the model to flag (not obey) embedded instructions via
`injection_attempt_detected`.

## Known limitations / what we'd do with more time

- Amendment currency conversion assumes the event's own settlement_date
  has an exchange rate row; a message stating an amount on a date with no
  corresponding row in `exchange_rates.csv` isn't handled.
- The extraction schema occasionally targets the same fact two different
  ways across near-duplicate runs (e.g. `event:event_2508` vs `unknown`
  for the same arrears payment) — harmless today (missing `date` on the
  `unknown` path means it's dropped, not double-counted), but worth
  tightening with a few-shot example in the prompt.
- `earliest_full_payment_date` is O(90) `simulate()` calls per request —
  fine at 250 rows, would want a smarter incremental scan at real scale.
- The trailing-consistency rate-change detection is a heuristic, not a
  certainty — on at least one sample request it reads a genuine rate
  change as sufficient on its own where the ground truth still expected
  a small additional spending cut. Reasonable interpretations can differ
  here; documented rather than silently "corrected" further.
- LLM calls run sequentially, not in parallel — fine at 250 rows (a few
  minutes), would want batching/concurrency at real scale.
