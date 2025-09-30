import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { parse } from 'csv-parse/sync';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.join(__dirname, '..', 'data');
const RESPONSE_PATH = path.join(DATA_DIR, 'responses.json');
const SUMMARY_CACHE_PATH = path.join(DATA_DIR, 'extracted_summaries.json');

function loadCsv(file) {
  const full = path.join(DATA_DIR, file);
  const content = fs.readFileSync(full, 'utf8');
  return parse(content, { columns: true, skip_empty_lines: true });
}

const FIELD_KEYWORDS = {
  population: [/population/i, /participants/i, /sample/i, /patients/i, /subjects/i],
  symptoms: [/symptom/i, /presentation/i, /clinical feature/i],
  riskFactors: [/risk/i, /trigger/i, /cause/i, /predictor/i],
  interventions: [/intervention/i, /treatment/i, /therapy/i, /strategy/i],
  outcomes: [/outcome/i, /result/i, /effect/i, /impact/i]
};

function stripMarkdown(text) {
  return (text || '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1')
    .replace(/[_#>]/g, '')
    .trim();
}

function parseExtractionSummary(raw) {
  const text = stripMarkdown(raw || '').replace(/\r/g, '').trim();
  const sections = {};
  const lines = text.split('\n');
  let current = 'Summary';
  sections[current] = [];
  for (const line of lines) {
    const trimmed = stripMarkdown(line);
    if (!trimmed) continue;
    const headingMatch = trimmed.match(/^[-*]?\s*\*\*(.+?)\*\*\s*:?.*$/);
    const hashMatch = trimmed.match(/^#{2,}\s*(.+?)\s*:?$/);
    if (headingMatch) {
      current = headingMatch[1].trim();
      if (!sections[current]) sections[current] = [];
      continue;
    }
    if (hashMatch) {
      current = hashMatch[1].trim();
      if (!sections[current]) sections[current] = [];
      continue;
    }
    if (!sections[current]) sections[current] = [];
    sections[current].push(trimmed.replace(/^[-*]\s*/, ''));
  }
  const normalized = Object.entries(sections).map(([heading, bullets]) => ({ heading, bullets }));
  return { raw: text, sections: normalized };
}

function categorizeExtraction(parsed) {
  const buckets = {
    population: [],
    symptoms: [],
    riskFactors: [],
    interventions: [],
    outcomes: [],
    other: []
  };

  const assign = (text, fallbackHeading = '') => {
    const normalised = `${fallbackHeading} ${text}`.toLowerCase();
    for (const [key, patterns] of Object.entries(FIELD_KEYWORDS)) {
      if (patterns.some(rx => rx.test(normalised))) {
        buckets[key].push(text.trim());
        return;
      }
    }
    buckets.other.push(text.trim());
  };

  (parsed.sections || []).forEach(section => {
    const heading = section.heading || '';
    const bullets = section.bullets || [];
    if (bullets.length === 0) {
      assign(heading, heading);
      return;
    }
    bullets.forEach(bullet => {
      // split "Label: value" forms
      const parts = bullet.split(/:\s*/);
      if (parts.length > 1 && parts[0].length < 80) {
        assign(parts.slice(1).join(': '), `${heading} ${parts[0]}`);
      } else {
        assign(bullet, heading);
      }
    });
  });

  const summaryPieces = [];
  ['population', 'symptoms', 'riskFactors', 'interventions', 'outcomes'].forEach(key => {
    const unique = Array.from(new Map(buckets[key].map(v => [v.toLowerCase(), v])).values()).slice(0, 2);
    buckets[key] = unique.map(v => v.length > 120 ? `${v.slice(0, 117)}…` : v);
    if (buckets[key].length) {
      summaryPieces.push(`${key.replace(/([A-Z])/g, ' $1')}: ${buckets[key][0]}`);
    }
  });
  if (!summaryPieces.length && buckets.other.length) {
    summaryPieces.push(buckets.other[0]);
  }

  return {
    population: buckets.population,
    symptoms: buckets.symptoms,
    riskFactors: buckets.riskFactors,
    interventions: buckets.interventions,
    outcomes: buckets.outcomes,
    summaryFallback: summaryPieces.join(' | ')
  };
}

const qaRaw = loadCsv('qa_pairs.csv');
const extractedRaw = loadCsv('extracted_insights.csv');
const structuredColumnsPresent = extractedRaw.length > 0 &&
  Object.prototype.hasOwnProperty.call(extractedRaw[0], 'population');

function cleanTitle(title = '') {
  return title.replace(/^\s*\[/, '').replace(/\]\s*$/, '').trim();
}

const qaItems = qaRaw.map((row, idx) => ({
  id: `${row.pmid || 'unknown'}-${idx}`,
  pmid: row.pmid,
  title: cleanTitle(row.title || ''),
  question: row.qa_question,
  answer: row.qa_answer,
  explanation: row.qa_explanation,
  qa_type: row.qa_type,
  journal: row.journal,
  year: row.year,
  abstract: row.abstract,
  classification: row.classification,
  sourceIndex: idx
}));

const qaByPmid = qaItems.reduce((acc, item) => {
  if (!acc[item.pmid]) acc[item.pmid] = [];
  acc[item.pmid].push(item);
  return acc;
}, {});

const extractedItems = extractedRaw.map((row, idx) => {
  const parsed = parseExtractionSummary(row.gpt_output);
  const categorizedFromMarkdown = categorizeExtraction(parsed);

  const parseField = (value) => {
    if (!value) return [];
    if (Array.isArray(value)) return value;
    return value.split('||').map(part => stripMarkdown(part)).map(s => s.trim()).filter(Boolean);
  };

  const categorized = structuredColumnsPresent ? {
    population: parseField(row.population),
    symptoms: parseField(row.symptoms),
    riskFactors: parseField(row.risk_factors || row.riskFactors),
    interventions: parseField(row.interventions),
    outcomes: parseField(row.outcomes)
  } : categorizedFromMarkdown;

  const summary = structuredColumnsPresent
    ? stripMarkdown(row.structured_summary || row.summary || categorized.summaryFallback)
    : categorized.summaryFallback;

  return {
  id: `${row.pmid || 'unknown'}-${idx}`,
  pmid: row.pmid,
  title: cleanTitle(row.title || ''),
  journal: row.journal,
  year: row.year,
  abstract: row.abstract,
  gpt_output: row.gpt_output,
  classification: row.classification,
  sourceIndex: idx,
  parsed,
  categorized,
  summary
};
});

const datasets = {
  qa: {
    key: 'qa',
    label: 'Q&A Validation',
    items: qaItems,
    byPmid: qaByPmid
  },
  extracted: {
    key: 'extracted',
    label: 'Extraction Validation',
    items: extractedItems
  }
};

const datasetIndex = Object.fromEntries(Object.values(datasets).map(ds => [
  ds.key,
  Object.fromEntries(ds.items.map(item => [item.id, item]))
]));

async function callAmplify(messages) {
  const apiKey = process.env.AMPLIFY_API_KEY;
  const baseUrl = process.env.AMPLIFY_API_URL;
  if (!apiKey || !baseUrl) return null;
  try {
    const headerName = process.env.AMPLIFY_HEADER_NAME || 'Authorization';
    const headers = { 'Content-Type': 'application/json' };
    if (headerName.toLowerCase() === 'authorization') {
      headers[headerName] = `Bearer ${apiKey}`;
    } else {
      headers[headerName] = apiKey;
    }
    const payload = baseUrl.replace(/\/$/, '').endsWith('/chat')
      ? { data: { messages, max_tokens: 250, temperature: 0, options: { model: { id: process.env.AMPLIFY_MODEL || 'gpt-4o-mini' }, skipRag: true }, dataSources: [] } }
      : { model: process.env.AMPLIFY_MODEL || 'gpt-4o-mini', messages, max_tokens: 250, temperature: 0 };
    const res = await fetch(baseUrl, { method: 'POST', headers, body: JSON.stringify(payload) });
    if (!res.ok) return null;
    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || data?.data?.output_text || JSON.stringify(data);
    return text;
  } catch (err) {
    console.error('Amplify summary failed', err);
    return null;
  }
}

async function enrichSummaries(items) {
  if (!items.length) return;
  let cache = {};
  if (fs.existsSync(SUMMARY_CACHE_PATH)) {
    try {
      cache = JSON.parse(fs.readFileSync(SUMMARY_CACHE_PATH, 'utf8'));
    } catch (err) {
      console.warn('Failed to parse summary cache', err);
    }
  }
  const limit = parseInt(process.env.CURATOR_SUMMARY_LIMIT || '40', 10);
  let updated = false;
  for (const item of items.slice(0, Math.max(limit, 1))) {
    if (cache[item.id]) {
      item.summary = cache[item.id].summary || item.summary;
      if (cache[item.id].categorized) {
        item.categorized = cache[item.id].categorized;
      }
      continue;
    }
    const prompt = `You curate concise clinical evidence. Based on the raw extraction notes below, respond with strict JSON: {"population":[],"symptoms":[],"risk_factors":[],"interventions":[],"outcomes":[],"summary":""}.\n- Each list must contain up to 2 distinct entries.\n- Each entry must be ≤20 words and answer what the abstract says about that category.\n- Use "Not reported" if the abstract provides no support.\n- The summary must be ≤40 words.\nRaw notes:\n${item.gpt_output}`;
    const response = await callAmplify([
      { role: 'system', content: 'You craft concise, structured medical summaries without speculation.' },
      { role: 'user', content: prompt }
    ]);
    if (response) {
      try {
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        const parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : JSON.parse(response);
        const sanitise = (val) => {
          if (!val) return [];
          if (Array.isArray(val)) return val.map(stripMarkdown).filter(Boolean);
          return [stripMarkdown(val)];
        };
        const dedupe = (arr) => Array.from(new Map(arr.map(v => [v.toLowerCase(), v])).values());
        const enriched = {
          population: dedupe(sanitise(parsed.population)),
          symptoms: dedupe(sanitise(parsed.symptoms)),
          riskFactors: dedupe(sanitise(parsed.risk_factors || parsed.riskFactors)),
          interventions: dedupe(sanitise(parsed.interventions)),
          outcomes: dedupe(sanitise(parsed.outcomes))
        };
        item.categorized = enriched;
        item.summary = stripMarkdown(parsed.summary || item.summary);
        cache[item.id] = { summary: item.summary, categorized: enriched };
        updated = true;
        continue;
      } catch (err) {
        console.warn('Failed to parse LLM response', err);
      }
    }
    cache[item.id] = { summary: item.summary, categorized: item.categorized };
    updated = true;
  }
  if (updated) {
    fs.writeFileSync(SUMMARY_CACHE_PATH, JSON.stringify(cache, null, 2), 'utf8');
  }
  items.forEach(item => {
    if (cache[item.id]) {
      item.summary = cache[item.id].summary || item.summary;
      item.categorized = cache[item.id].categorized || item.categorized;
    }
  });
}

const needsEnrichment = !structuredColumnsPresent && process.env.CURATOR_USE_SUMMARY !== '0';
if (needsEnrichment) {
  await enrichSummaries(extractedItems);
}

function normalizeStore(store = {}) {
  const normalised = {
    responses: store.responses || {},
    compare: store.compare || {},
    reviewedItems: store.reviewedItems || {}
  };
  for (const [questionId, payload] of Object.entries(normalised.responses)) {
    if (!payload) {
      normalised.responses[questionId] = { counts: {}, records: [] };
      continue;
    }
    const hasNewShape = Object.prototype.hasOwnProperty.call(payload, 'records');
    if (!hasNewShape) {
      const counts = {};
      for (const [key, value] of Object.entries(payload)) {
        if (typeof value === 'number') counts[key] = value;
      }
      normalised.responses[questionId] = { counts, records: [] };
    } else {
      payload.counts = payload.counts || {};
      payload.records = payload.records || [];
    }
  }
  return normalised;
}

function loadResponses() {
  if (!fs.existsSync(RESPONSE_PATH)) {
    return {
      qa: { responses: {}, compare: {}, reviewedItems: {} },
      extracted: { responses: {}, compare: {}, reviewedItems: {} }
    };
  }
  try {
    const raw = fs.readFileSync(RESPONSE_PATH, 'utf8');
    const parsed = JSON.parse(raw);
    return {
      qa: normalizeStore(parsed.qa),
      extracted: normalizeStore(parsed.extracted)
    };
  } catch (err) {
    console.error('Failed to load response store', err);
    return {
      qa: { responses: {}, compare: {}, reviewedItems: {} },
      extracted: { responses: {}, compare: {}, reviewedItems: {} }
    };
  }
}

let responseStore = loadResponses();

function saveResponses() {
  fs.writeFileSync(RESPONSE_PATH, JSON.stringify(responseStore, null, 2), 'utf8');
}

function ensureQuestionStore(store, questionId) {
  if (!store.responses[questionId]) {
    store.responses[questionId] = {
      counts: {},
      records: []
    };
  }
  return store.responses[questionId];
}

function aggregateSummary(datasetKey) {
  const store = responseStore[datasetKey];
  if (!store) {
    return { totalDecisions: 0, reviewed: 0, questionSummaries: {}, compare: {} };
  }
  const questionSummaries = {};
  let totalDecisions = 0;
  for (const [questionId, payload] of Object.entries(store.responses)) {
    const counts = payload.counts || {};
    totalDecisions += Object.values(counts).reduce((a, b) => a + b, 0);
    const yes = payload.records.filter(r => r.answer === 'yes');
    const no = payload.records.filter(r => r.answer === 'no');
    questionSummaries[questionId] = {
      counts,
      yes,
      no
    };
  }
  const reviewed = Object.keys(store.reviewedItems || {}).length;
  return {
    totalDecisions,
    reviewed,
    questionSummaries,
    compare: store.compare || {}
  };
}

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/api/datasets', (_req, res) => {
  const list = Object.values(datasets).map(ds => ({
    key: ds.key,
    label: ds.label,
    count: ds.items.length,
    uniquePmids: new Set(ds.items.map(it => it.pmid)).size
  }));
  res.json({ datasets: list });
});

app.get('/api/items', (req, res) => {
  const { dataset: datasetKey = 'qa', offset = '0', limit = '20', q = '' } = req.query;
  const dataset = datasets[datasetKey];
  if (!dataset) return res.status(400).json({ error: 'Unknown dataset' });
  const search = (q || '').toString().trim().toLowerCase();
  let filtered = dataset.items;
  if (search) {
    filtered = dataset.items.filter(item => {
      return Object.values(item).some(val =>
        typeof val === 'string' && val.toLowerCase().includes(search)
      );
    });
  }
  const start = Math.max(parseInt(offset, 10) || 0, 0);
  const end = start + (Math.max(parseInt(limit, 10) || 20, 1));
  const slice = filtered.slice(start, end);
  res.json({
    items: slice,
    total: filtered.length,
    offset: start,
    limit: end - start
  });
});

app.get('/api/compare-options', (req, res) => {
  const { dataset: datasetKey = 'qa', pmid } = req.query;
  const dataset = datasets[datasetKey];
  if (!dataset) return res.status(400).json({ error: 'Unknown dataset' });
  if (!pmid) return res.status(400).json({ error: 'pmid required' });
  const list = (dataset.byPmid && dataset.byPmid[pmid]) || [];
  res.json({ items: list });
});

app.post('/api/responses', (req, res) => {
  const { dataset: datasetKey, itemId, questionId, answer, sure } = req.body || {};
  if (!datasetKey || !datasets[datasetKey]) return res.status(400).json({ error: 'Unknown dataset' });
  if (!itemId || !questionId) return res.status(400).json({ error: 'Missing item/question id' });
  if (!['yes', 'no'].includes(answer)) return res.status(400).json({ error: 'Answer must be yes or no' });
  const store = responseStore[datasetKey];
  const item = datasetIndex[datasetKey][itemId];
  if (!item) return res.status(404).json({ error: 'Item not found' });

  const questionStore = ensureQuestionStore(store, questionId);
  const key = `${answer}_${sure ? 'sure' : 'unsure'}`;
  questionStore.counts[key] = (questionStore.counts[key] || 0) + 1;

  const record = {
    timestamp: new Date().toISOString(),
    dataset: datasetKey,
    questionId,
    answer,
    sure: Boolean(sure),
    itemId,
    pmid: item.pmid,
    title: item.title,
    journal: item.journal,
    year: item.year,
    classification: item.classification,
    content: datasetKey === 'qa'
      ? { question: item.question, answer: item.answer, explanation: item.explanation }
      : { summary: item.summary, fields: item.categorized, rawSections: item.parsed.sections }
  };
  questionStore.records.push(record);

  store.reviewedItems[itemId] = store.reviewedItems[itemId] || {};
  store.reviewedItems[itemId][questionId] = record;

  saveResponses();
  res.json({ ok: true });
});

app.post('/api/compare', (req, res) => {
  const { dataset: datasetKey = 'qa', pmid, choiceId } = req.body || {};
  if (!datasets[datasetKey]) return res.status(400).json({ error: 'Unknown dataset' });
  if (!pmid || !choiceId) return res.status(400).json({ error: 'pmid and choice required' });
  const store = responseStore[datasetKey];
  if (!store.compare[pmid]) store.compare[pmid] = {};
  store.compare[pmid][choiceId] = (store.compare[pmid][choiceId] || 0) + 1;
  saveResponses();
  res.json({ ok: true });
});

app.get('/api/summary', (req, res) => {
  const { dataset: datasetKey = 'qa' } = req.query;
  if (!datasets[datasetKey]) return res.status(400).json({ error: 'Unknown dataset' });
  const summary = aggregateSummary(datasetKey);
  res.json(summary);
});

app.get('/api/records', (req, res) => {
  const { dataset: datasetKey = 'qa', questionId, decision } = req.query;
  if (!datasets[datasetKey]) return res.status(400).json({ error: 'Unknown dataset' });
  if (!questionId) return res.status(400).json({ error: 'questionId required' });
  const store = responseStore[datasetKey];
  const questionStore = (store.responses || {})[questionId];
  if (!questionStore) return res.json({ records: [] });
  let records = questionStore.records || [];
  if (decision === 'yes' || decision === 'no') {
    records = records.filter(r => r.answer === decision);
  }
  res.json({ records });
});

function toCsv(rows) {
  if (!rows.length) return '';
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push(headers.map(h => {
      const val = row[h] == null ? '' : String(row[h]);
      if (val.includes(',') || val.includes('"') || val.includes('\n')) {
        return '"' + val.replace(/"/g, '""') + '"';
      }
      return val;
    }).join(','));
  }
  return lines.join('\n');
}

app.get('/api/export', (req, res) => {
  const { dataset: datasetKey = 'qa', questionId, decision, format = 'csv' } = req.query;
  if (!datasets[datasetKey]) return res.status(400).json({ error: 'Unknown dataset' });
  const store = responseStore[datasetKey];
  const questions = store.responses || {};
  let records = [];

  const pushRecords = (qid) => {
    const questionStore = questions[qid];
    if (!questionStore) return;
    let recs = questionStore.records || [];
    if (decision === 'yes' || decision === 'no') {
      recs = recs.filter(r => r.answer === decision);
    }
    records = records.concat(recs.map(r => ({
      timestamp: r.timestamp,
      dataset: r.dataset,
      questionId: r.questionId,
      decision: r.answer,
      sure: r.sure,
      pmid: r.pmid,
      title: r.title,
      journal: r.journal,
      year: r.year,
      classification: r.classification,
      summary: r.dataset === 'qa'
        ? `${r.content.question} | ${r.content.answer}`
        : `${r.content.summary || ''}`
    })));
  };

  if (questionId) {
    pushRecords(questionId);
  } else {
    Object.keys(questions).forEach(pushRecords);
  }

  if (format === 'json') {
    res.json({ records });
    return;
  }

  const csv = toCsv(records);
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="${datasetKey}-responses.csv"`);
  res.send(csv);
});

const port = process.env.PORT || 3400;
app.listen(port, () => {
  console.log(`knowledge-curator-v1 running at http://localhost:${port}`);
});
