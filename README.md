# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon (September 2026).

## Buy or Wait?

Build an AI-powered financial agent that decides whether a user can safely afford a requested expense.

A user may ask: **"Can I afford this laptop?"**

Answering well takes more than the current balance. The agent must account for recurring expenses, pending payments, essential spending, confirmed income, available payment options, and relevant details buried in messages and images.

For every request, the agent decides whether the user should pay in full, pay partially, use installments, wait, or not proceed. The recommendation must be personalized: two users with the same balance can deserve different answers based on their commitments, priorities, payment preferences, and willingness to adjust flexible expenses.

A recommendation is safe only if the user can complete the full payment plan, cover essential expenses, and stay above their preferred minimum balance throughout the forecast period.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, conflict-resolution rules, and submission format.

---

## Quick Start

Clone the repository and move into the project directory:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-september26.git
cd hackerrank-orchestrate-september26
```

Build your solution in `code/main.py`, or use another language and document its entry point clearly.

Your solution must:

- Read the input files from `dataset/`
- Generate one prediction for every request
- Write the final predictions to `output.csv` in the repository root

Run the starter Python entry point with:

```bash
python3 code/main.py
```

After running your solution, confirm that `output.csv` exists in the repository root and contains the required columns and one row for every request.

## Important File Locations

```text
dataset/        Input data and the blank output template. Do not modify the input data.
code/           Your solution code.
output.csv      Final generated predictions in the repository root.
code.zip        ZIP file containing your complete solution for submission.
```

The blank template at `dataset/output.csv` is provided as a reference. Your final generated file must be the root-level `output.csv`.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
├── code/                             # Your solution code
├── output.csv                        # Final generated predictions
└── dataset/
    ├── requests.csv                  # 250 requests to evaluate — predict these
    ├── output.csv                    # Blank submission template
    ├── sample_requests.csv           # 25 solved examples
    ├── financial_profiles.csv        # Balances, minimum balance, priorities, preferences
    ├── financial_events.csv          # Historical, pending, and confirmed transactions
    ├── request_payment_options.csv   # Payment options available per request
    ├── exchange_rates.csv            # Fixed, dated conversion rates
    ├── messages.csv                  # Messages tied to users, requests, or events
    ├── images.csv                    # Payroll letters, statements, bills, receipts
    └── media/
        └── images/
```

Only `dataset/requests.csv` requires predictions. Everything else is context. Join user records with `user_id`, request records with `request_id`, supporting evidence with `related_event_id`, and exchange rates with the rate date and currency pair.

Amounts are in the user's `home_currency` — the dataset uses INR, ZAR, IDR, USD, and EUR, and every conversion rate you need is in `exchange_rates.csv`. All dates are `YYYY-MM-DD`. Live exchange rates, market data, and banking access are not required.

---

## What You Need to Build

For every row in `dataset/requests.csv`, produce one row in `output.csv` with:

| Column | Meaning |
|---|---|
| `request_id` | The request being answered |
| `amount_safe_to_pay` | Largest amount safe to pay on `request_date` before optional spending changes, after protecting essentials and the minimum balance |
| `affordability_status` | `affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable` |
| `recommended_payment_method` | `full_payment`, `partial_payment`, `installments`, `wait`, or `not_recommended` |
| `payment_plan` | Chronological `<YYYY-MM-DD>:<amount>` entries joined by `\|`, or `none` |
| `earliest_date_for_full_payment` | Earliest date the full amount is forecast safe as one payment; empty if never within the forecast |
| `spending_changes_needed` | Up to three `stop:<event_id>` / `reduce_to:<event_id>:<amount>` changes joined by `\|`, or `none` |
| `decision_explanation` | Short explanation and the financial facts behind it |

`0 <= amount_safe_to_pay <= requested_amount` must always hold. Installment plans must exactly match a supplied payment option, and only recurring expenses marked flexible may be changed.

`affordable_with_plan` means the full request is completed through a partial-payment schedule, installments, or permitted spending changes. Recommend `partial_payment` only when the request allows it, the user accepts it, `0 < amount_safe_to_pay < requested_amount`, and `earliest_date_for_full_payment` is on or before `desired_completion_date`. Use exactly two payments: pay `amount_safe_to_pay` on `request_date`, then pay the remaining amount on `earliest_date_for_full_payment`. The two payments must add up to `requested_amount`. Unlike installments, partial payment does not need to match a supplied payment option.

---

## Suggested Workflow

1. Inspect `dataset/sample_requests.csv` — 25 requests with completed output columns — to understand the expected format and decision style.
2. Reconstruct each user's financial state from `financial_profiles.csv` and `financial_events.csv`: separate recurring expenses from one-time events, reserve pending transactions, count confirmed salary only on its settlement date, and de-duplicate repeated representations of the same event.
3. When an event has a blank `amount`, find its `event_id` as `related_event_id` in `images.csv` and extract the amount from the linked image. Never treat a blank amount as zero. Pull in any other relevant messages, images, and payment options for the request.
4. Forecast forward and generate a plan that keeps the balance above the minimum at every step.
5. Verify deterministically — bounds, plan feasibility, schedule match, flexible-only spending changes — before writing `output.csv`.
6. Score yourself on the solved samples, then run the full dataset.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv` with the exact required columns in the exact required order
- include one prediction for every `request_id` in `dataset/requests.csv`
- not use organizer-only files or hardcoded labels
- keep behavior deterministic where possible

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth values.

The scoring will consider:

- accuracy of `amount_safe_to_pay`
- correctness of `affordability_status`
- correctness of `recommended_payment_method` and `payment_plan`
- accuracy of `earliest_date_for_full_payment`
- validity of `spending_changes_needed`
- usefulness and consistency of `decision_explanation`

### Token Usage And Cost Analysis

Your `code.zip` must include one token-usage file:

```text
evaluation/usage_report.md
```

The report must cover model providers and names, model calls, input and output tokens, total and average tokens per request, estimated total and per-request cost. The reported values must correspond to the final full-dataset run that produced your `output.csv`.

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to a `log.txt` in the repository root — the same directory as `AGENTS.md`:

| Platform | Path |
|---|---|
| macOS / Linux | `<repo root>/log.txt` |
| Windows | `<repo root>\log.txt` |

The path resolves relative to `AGENTS.md`, so it stays correct across clones, renames, and checkouts. `log.txt` is gitignored — upload it as your chat transcript at submission time. Do not paste secrets into the chat.

In case, the harness you are using is not in the repo root, you can explicitly ask the agent to look for the AGENTS.md in this folder & then continue.

---

## Submission

Submit the following files as instructed by HackerRank:

| File | Description |
|---|---|
| `code.zip` | Full runnable solution, prompts/configuration, README, and the required `evaluation/` folder |
| `output.csv` | Predictions for every row in `dataset/requests.csv` |
| `chat_transcript` | The `log.txt` described above, showing how you developed or used the system |

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/requests.csv` (250 rows plus the header).
- `output.csv` has the exact required columns in the exact required order.
- Every `amount_safe_to_pay` satisfies `0 <= amount_safe_to_pay <= requested_amount`.
- Every installment plan matches a supplied payment option, and every spending change targets a flexible recurring expense.
- Your runnable code, setup instructions, and `evaluation/` folder are included in `code.zip`.

---
# Buy or Wait? — Solution README

An AI-powered financial affordability agent for HackerRank Orchestrate
(September 2026). For each request in `dataset/requests.csv`, decides
whether the user can safely pay in full, pay partially, use an
installment plan, wait, or shouldn't proceed at all — grounded in a
90-day balance forecast, not a single-number balance check.

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
                     installments / partial_payment), filters by what the
                     user will accept, ranks by the spec's 6-rule order
main.py           -- orchestrates the above per request, writes output.csv
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
recover a blank amount. That's the only piece we hand to Claude.

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

**LLM calls are batched per user, not per message.** Checked the data
first: every user has at most one message and at most one image, and
`requests.csv` maps users to requests 1:1. So "per user" and "per
request" extraction are the same granularity here — and ~31 of 250 users
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

- `spending_changes_needed` (stop:/reduce_to:) is not implemented yet —
  candidates never model an optional spending change, so `planner.py`
  can miss an `affordable_with_plan` outcome that only becomes safe after
  trimming a flexible expense.
- `decision_explanation` is factual but terse — not natural-language.
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

## Setup & running

```bash
pip install pandas anthropic python-dotenv
cp .env.example .env   # fill in your real ANTHROPIC_API_KEY
python3 code/main.py
```

Runs against `dataset/` by default and writes `output.csv` to the repo
root plus `evaluation/usage_report.md`. Use `--limit N` to test on the
first N requests without spending on the full run. Works without an API
key too — the LLM step is skipped gracefully and the rest of the
pipeline (currency, forecast, planner) runs unchanged.

Model: `claude-haiku-4-5-20251001` — the cheapest current Claude model,
sufficient here because extraction is short-text/single-image →
fixed-schema JSON, not open-ended reasoning. See
`evaluation/usage_report.md` for the actual token/cost numbers from the
run that produced `output.csv`.
