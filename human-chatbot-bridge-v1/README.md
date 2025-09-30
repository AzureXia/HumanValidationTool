# Human↔Chatbot Bridge v1

Utility scripts that convert reviewer-approved knowledge from **Human Expert Validation v1** (Track B) into curated datasets consumable by the **medical-chatbot-v0** experience (Track A).

## Why this exists
- Keep each agent independent: reviewers work in `human-expert-validation-v1`, end users chat through `medical-chatbot-v0`.
- When a review cycle finishes, run this bridge to generate fresh, trusted JSONL files for the chatbot without modifying either project.

## What it does
1. Loads reviewer responses from `../human-expert-validation-v1/data/responses.json`.
2. Rehydrates the source items from the CSVs shipped with the reviewer tool.
3. Extracts only the entries marked **YES** for both QA validity and QA quality (or extraction accuracy).
4. Writes two newline-delimited JSON files under `../medical-chatbot-v0/datasets/validated/`:
   - `qa.jsonl` — validated question/answer pairs with metadata.
   - `extractions.jsonl` — validated structured findings (population, symptoms, risk factors, interventions, outcomes).
5. Generates a small `latest_export.json` summary (counts, timestamp, source paths).

Nothing inside the reviewer or chatbot folders is altered beyond those generated files.

## Quick Start

```bash
cd assignment5/human-chatbot-bridge-v1
python3 export_validated.py
```

Outputs:
```
medical-chatbot-v0/datasets/validated/
├── qa.jsonl
├── extractions.jsonl
└── latest_export.json
```

Use these files when indexing documents for the medical chatbot (e.g., load them in `enhanced_rag.py`).

### No reviewer approvals yet?

You can still populate demo datasets that pretend every row is approved:

```bash
cd assignment5/human-chatbot-bridge-v1
python3 prepare_demo_validated.py
```

This uses `trial_data_by_ass3_agent/generated_qa_984rows.csv` and `trial_data_by_ass3_agent/step3_extracted_larger.csv` to create the same JSONL files.

## Workflow Suggestion
1. Review/approve knowledge in `human-expert-validation-v1`.
2. Run `python3 export_validated.py` from this bridge folder.
3. Rebuild the chatbot’s retrieval index using the generated JSONL files.
4. Demo Track B (builder view), then immediately showcase Track A (end-user chatbot) using the refreshed dataset.

## Implementation Notes
- Script uses only Python’s standard library (no extra requirements).
- IDs mirror the reviewer tool (`{pmid}-{row_index}`), so future cross-referencing stays consistent.
- If `responses.json` is missing, the script exits gracefully with an informative message.
- Existing exports are overwritten atomically on each run.
