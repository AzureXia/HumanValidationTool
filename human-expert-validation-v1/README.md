# Human Expert Validation v1

Human-in-the-loop review UI for Assignment&nbsp;3 outputs. Mirrors the look and feel of the Assignment&nbsp;3 interactive demo, but adds structured validation workflows for domain experts.

## Features

- Load large QA (`step4`) and extraction (`step3`) CSVs (provided in `data/`).
- Card-based review with highlighted article metadata and abstract context.
- Structured evaluations with confirmation modal (`Yes, confident` / `No, unsure`).
- Real-time progress tracker (`reviewed` vs total) and rich summary dashboard.
- QA comparison modal: pick the strongest question per abstract.
- Full audit trail kept per response (timestamp, item snapshot, reviewer confidence).
- Export buttons for CSV/JSON downloads (per question or entire session).

## Quick Start

```bash
cd assignment5/human-expert-validation-v1
cp .env.example .env           # optional – Amplify vars included for parity
# Optional: enable Amplify-powered summaries (defaults to on if creds exist)
# export CURATOR_USE_SUMMARY=1
npm install
npm run dev                    # server on http://localhost:3400
```

### Refresh extraction dataset with Amplify (optional)

```bash
cd assignment5/human-expert-validation-v1
# ensure .env has AMPLIFY_API_KEY / AMPLIFY_API_URL
node scripts/build_extractions.mjs      # creates data/extracted_insights.csv using up to 80 abstracts
```

Open the browser to review items. Use the sidebar to switch datasets, search, and view summaries. Responses are recorded in `data/responses.json` (created on first submission).

## API

| Endpoint | Description |
|---|---|
| `GET /api/datasets` | Dataset metadata (counts, unique PMIDs).
| `GET /api/items?dataset=qa&offset=0&limit=10&q=` | Paginated items.
| `POST /api/responses` | Store validation: `{ dataset, itemId, questionId, answer, sure }`.
| `POST /api/compare` | Record selected QA pair: `{ dataset, pmid, choiceId }`.
| `GET /api/compare-options?dataset=qa&pmid=` | QA pairs for comparison.
| `GET /api/summary?dataset=qa` | Aggregated counts, recent examples, totals, comparisons.
| `GET /api/records?dataset=qa&questionId=&decision=` | Raw records for a given question (filter by decision optional).
| `GET /api/export?dataset=qa&questionId=&decision=&format=csv` | Download responses as CSV/JSON (per question or all).

Feel free to swap in new CSVs: drop them into `data/` and adjust filenames in `src/server.js`.
