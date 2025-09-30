#!/usr/bin/env python3
"""Bridge reviewer-approved data into chatbot datasets."""

import csv
import json
import os
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]
REVIEWER_DIR = BASE / 'human-expert-validation-v1' / 'data'
CHATBOT_DIR = BASE / 'medical-chatbot-v0' / 'datasets' / 'validated'

QA_CSV = REVIEWER_DIR / 'qa_pairs.csv'
EXTRACT_CSV = REVIEWER_DIR / 'extracted_insights.csv'
RESPONSES_JSON = REVIEWER_DIR / 'responses.json'


def load_csv_index(path):
    """Load CSV rows and return {item_id: row_dict}."""
    if not path.exists():
        raise FileNotFoundError(f'Missing data file: {path}')
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    index = {}
    for idx, row in enumerate(rows):
        pmid = row.get('pmid') or 'unknown'
        item_id = f"{pmid}-{idx}"
        index[item_id] = row
    return index


def load_responses():
    if not RESPONSES_JSON.exists():
        raise FileNotFoundError('No reviewer responses yet. Run the reviewer tool first.')
    with RESPONSES_JSON.open() as f:
        return json.load(f)


def ensure_output_dir():
    CHATBOT_DIR.mkdir(parents=True, exist_ok=True)


def export_jsonl(path, records):
    with path.open('w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    qa_index = load_csv_index(QA_CSV)
    extracted_index = load_csv_index(EXTRACT_CSV)
    responses = load_responses()

    qa_exports = []
    qa_reviewed = responses.get('qa', {}).get('reviewedItems', {})
    for item_id, judgements in qa_reviewed.items():
        reflects = judgements.get('qa_reflects')
        quality = judgements.get('qa_quality')
        if not reflects or not quality:
            continue
        if reflects.get('answer') != 'yes' or quality.get('answer') != 'yes':
            continue
        item = qa_index.get(item_id)
        if not item:
            continue
        qa_exports.append({
            'doc_id': f"{item.get('pmid') or item_id}-qa",
            'type': 'qa',
            'pmid': item.get('pmid'),
            'title': item.get('title'),
            'question': item.get('qa_question'),
            'answer': item.get('qa_answer'),
            'explanation': item.get('qa_explanation'),
            'abstract': item.get('abstract'),
            'journal': item.get('journal'),
            'year': item.get('year'),
            'classification': item.get('classification'),
            'confidence': {
                'reflects_sure': bool(reflects.get('sure')),
                'quality_sure': bool(quality.get('sure'))
            },
            'recorded_at': reflects.get('timestamp') or quality.get('timestamp')
        })

    extraction_exports = []
    extraction_reviewed = responses.get('extracted', {}).get('reviewedItems', {})
    for item_id, judgements in extraction_reviewed.items():
        accuracy = judgements.get('extracted_accuracy')
        if not accuracy or accuracy.get('answer') != 'yes':
            continue
        item = extracted_index.get(item_id)
        if not item:
            continue
        extraction_exports.append({
            'doc_id': f"{item.get('pmid') or item_id}-extraction",
            'type': 'extraction',
            'pmid': item.get('pmid'),
            'title': item.get('title'),
            'journal': item.get('journal'),
            'year': item.get('year'),
            'summary': item.get('summary') or item.get('structured_summary') or '',
            'population': (item.get('population') or '').split('||') if item.get('population') else [],
            'symptoms': (item.get('symptoms') or '').split('||') if item.get('symptoms') else [],
            'riskFactors': (item.get('risk_factors') or '').split('||') if item.get('risk_factors') else [],
            'interventions': (item.get('interventions') or '').split('||') if item.get('interventions') else [],
            'outcomes': (item.get('outcomes') or '').split('||') if item.get('outcomes') else [],
            'abstract': item.get('abstract'),
            'recorded_at': accuracy.get('timestamp'),
            'confidence': { 'accuracy_sure': bool(accuracy.get('sure')) }
        })

    ensure_output_dir()
    export_jsonl(CHATBOT_DIR / 'qa.jsonl', qa_exports)
    export_jsonl(CHATBOT_DIR / 'extractions.jsonl', extraction_exports)

    summary = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'inputs': {
            'qa_csv': str(QA_CSV),
            'extracted_csv': str(EXTRACT_CSV),
            'responses_json': str(RESPONSES_JSON)
        },
        'outputs': {
            'qa': str(CHATBOT_DIR / 'qa.jsonl'),
            'extractions': str(CHATBOT_DIR / 'extractions.jsonl')
        },
        'counts': {
            'qa': len(qa_exports),
            'extractions': len(extraction_exports)
        }
    }
    with (CHATBOT_DIR / 'latest_export.json').open('w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as err:
        print(f'[bridge] {err}')
        raise SystemExit(1)
