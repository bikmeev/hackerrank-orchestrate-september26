# Token Usage and Cost Report

## Run summary
- Requests processed: 250
- Model: claude-haiku-4-5-20251001 (Anthropic)
- LLM calls made: 192
- Skipped (no message/image evidence for that user): 50
- Skipped (no ANTHROPIC_API_KEY set): 0
- Errors (call failed, fell back to no-correction): 5

## Tokens
- Total input tokens: 231,854
- Total output tokens: 23,358
- Total tokens: 255,212
- Average tokens per LLM call: 1,329

## Cost (estimated — verify current Haiku 4.5 rates before submitting)
- Assumed rate: $1.00/MTok in, $5.00/MTok out
- Estimated total cost: $0.3486
- Estimated average cost per LLM call: $0.00182
- Estimated cost per request in requests.csv: $0.00139

## Notes
- One call per user (not per message/event) — see llm_extract.py module
  docstring for why that's the natural granularity for this dataset.
- Results are cached in .cache/llm_extractions.json across runs; only the
  first run pays for extraction, later runs during development are free.
