#!/usr/bin/env python3
"""Create demo validated datasets from QA and extraction CSVs."""

import csv
import json
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE / 'trial_data_by_ass3_agent'
QA_SOURCE = DATA_ROOT / 'generated_qa_984rows.csv'
EXTRACTION_SOURCE = DATA_ROOT / 'step3_extracted_larger.csv'
OUTPUT_DIR = BASE / 'medical-chatbot-v0' / 'datasets' / 'validated'


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return list(csv.DictReader(f))


def write_jsonl(path, records):
    with path.open('w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_qa(records, limit=200):
    docs = []
    for idx, row in enumerate(records[:limit]):
        pmid = row.get('pmid') or f'DEMO_QA_{idx:04d}'
        docs.append({
            'doc_id': f'{pmid}-qa',
            'type': 'qa',
            'pmid': pmid,
            'title': row.get('title'),
            'question': row.get('qa_question'),
            'answer': row.get('qa_answer'),
            'explanation': row.get('qa_explanation'),
            'abstract': row.get('abstract'),
            'journal': row.get('journal'),
            'year': row.get('year'),
            'classification': row.get('classification'),
            'confidence': {'reflects_sure': True, 'quality_sure': True}
        })
    return docs


def parse_sections(text):
    text = (text or '').replace('\r', '')
    sections = {}
    current = 'Summary'
    sections[current] = []
    for line in text.split('\n'):
        stripped = line.strip('* ').strip()
        if not stripped:
            continue
        if stripped.endswith(':') and len(stripped.split()) <= 5:
            current = stripped[:-1].strip() or current
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(stripped)
    return sections


def bucket_extraction(row):
    sections = parse_sections(row.get('gpt_output'))
    buckets = {'population': [], 'symptoms': [], 'riskFactors': [], 'interventions': [], 'outcomes': []}
    for heading, items in sections.items():
        key = heading.lower()
        target = None
        if any(term in key for term in ['population', 'participant', 'patient']):
            target = 'population'
        elif any(term in key for term in ['symptom', 'presentation']):
            target = 'symptoms'
        elif any(term in key for term in ['risk', 'trigger', 'cause']):
            target = 'riskFactors'
        elif any(term in key for term in ['intervention', 'treatment', 'therapy']):
            target = 'interventions'
        elif any(term in key for term in ['outcome', 'effect', 'result']):
            target = 'outcomes'
        if target:
            for item in items:
                cleaned = item.split(':', 1)[-1].strip() if ':' in item else item
                if cleaned and cleaned.lower() not in [e.lower() for e in buckets[target]]:
                    buckets[target].append(cleaned)
    return buckets


def build_extractions(records, limit=200):
    docs = []
    for idx, row in enumerate(records[:limit]):
        pmid = row.get('pmid') or f'DEMO_EXT_{idx:04d}'
        buckets = bucket_extraction(row)
        docs.append({
            'doc_id': f'{pmid}-extraction',
            'type': 'extraction',
            'pmid': pmid,
            'title': row.get('title'),
            'journal': row.get('journal'),
            'year': row.get('year'),
            'summary': row.get('summary', ''),
            'population': buckets['population'],
            'symptoms': buckets['symptoms'],
            'riskFactors': buckets['riskFactors'],
            'interventions': buckets['interventions'],
            'outcomes': buckets['outcomes'],
            'abstract': row.get('abstract'),
            'confidence': {'accuracy_sure': True}
        })
    return docs


def main():
    qa_rows = read_csv(QA_SOURCE)
    ext_rows = read_csv(EXTRACTION_SOURCE)

    qa_docs = build_qa(qa_rows)
    extraction_docs = build_extractions(ext_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / 'qa.jsonl', qa_docs)
    write_jsonl(OUTPUT_DIR / 'extractions.jsonl', extraction_docs)

    summary = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'qa_count': len(qa_docs),
        'extractions_count': len(extraction_docs),
        'source_files': {
            'qa': str(QA_SOURCE),
            'extractions': str(EXTRACTION_SOURCE)
        },
        'output_dir': str(OUTPUT_DIR)
    }
    with (OUTPUT_DIR / 'latest_export.json').open('w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
