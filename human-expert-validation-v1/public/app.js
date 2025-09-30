const state = {
  dataset: 'qa',
  page: 0,
  limit: 10,
  query: '',
  summary: null
};

const escapeHtml = (str = '') => str
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');

const evalQuestions = {
  qa: [
    {
      id: 'qa_reflects',
      text: 'Is this question correctly grounded in the abstract’s findings?'
    },
    {
      id: 'qa_quality',
      text: 'Does this QA pair stand alone as a high-value clinical knowledge statement?'
    }
  ],
  extracted: [
    {
      id: 'extracted_accuracy',
      text: 'Does this extracted summary accurately represent the abstract’s findings?'
    }
  ]
};

let datasets = [];
let pendingResponse = null;
let compareSelection = null;
let searchTimer = null;

const datasetListEl = document.getElementById('dataset-list');
const itemsEl = document.getElementById('items');
const datasetTitleEl = document.getElementById('dataset-title');
const pageInfoEl = document.getElementById('page-info');
const searchInput = document.getElementById('search-input');
const statsEl = document.getElementById('stats');
const progressEl = document.getElementById('progress');

const modal = document.getElementById('modal');
const modalQuestion = document.getElementById('modal-question');
const modalSubtext = document.getElementById('modal-subtext');
const modalConfirm = document.getElementById('modal-confirm');
const modalCancel = document.getElementById('modal-cancel');

const compareModal = document.getElementById('compare-modal');
const compareListEl = document.getElementById('compare-list');
const compareSubmit = document.getElementById('compare-submit');
const compareClose = document.getElementById('compare-close');

const summaryModal = document.getElementById('summary-modal');
const summaryBtn = document.getElementById('summary-btn');
const summaryClose = document.getElementById('summary-close');
const summaryContent = document.getElementById('summary-content');
const exportButtons = document.getElementById('export-buttons');

const toast = document.getElementById('toast');

async function init() {
  await fetchDatasets();
  await refreshSummary();
  await loadItems();
}

async function fetchDatasets() {
  const res = await fetch('/api/datasets');
  const data = await res.json();
  datasets = data.datasets || [];
  renderDatasetList();
  updateStats();
}

function renderDatasetList() {
  datasetListEl.innerHTML = '';
  datasets.forEach(ds => {
    const div = document.createElement('div');
    div.className = `dataset-card ${state.dataset === ds.key ? 'active' : ''}`;
    div.innerHTML = `<h3>${ds.label}</h3><p>${ds.count} items · ${ds.uniquePmids} studies</p>`;
    div.onclick = async () => {
      state.dataset = ds.key;
      state.page = 0;
      state.query = '';
      searchInput.value = '';
      renderDatasetList();
      updateStats();
      await refreshSummary();
      loadItems();
    };
    datasetListEl.appendChild(div);
  });
}

function updateStats() {
  const ds = datasets.find(d => d.key === state.dataset);
  if (!ds) return;
  statsEl.innerHTML = `
    <div><strong>Total Records:</strong> ${ds.count}</div>
    <div><strong>Unique Studies:</strong> ${ds.uniquePmids}</div>
  `;
  datasetTitleEl.textContent = ds.label;
  updateProgressBar();
}

function updateProgressBar() {
  const summary = state.summary;
  const ds = datasets.find(d => d.key === state.dataset);
  if (!summary || !ds) {
    progressEl.innerHTML = '';
    return;
  }
  const pct = Math.min(100, Math.round((summary.reviewed / Math.max(1, ds.count)) * 100));
  progressEl.innerHTML = `
    <div><strong>Reviewed:</strong> ${summary.reviewed} / ${ds.count} (${pct}%)</div>
    <div class="progress-bar"><span style="width:${pct}%"></span></div>
  `;
}

async function refreshSummary() {
  const params = new URLSearchParams({ dataset: state.dataset });
  const res = await fetch(`/api/summary?${params.toString()}`);
  const data = await res.json();
  state.summary = data;
  updateProgressBar();
}

async function loadItems() {
  const params = new URLSearchParams({
    dataset: state.dataset,
    offset: state.page * state.limit,
    limit: state.limit,
    q: state.query
  });
  const res = await fetch(`/api/items?${params.toString()}`);
  const data = await res.json();
  renderItems(data.items || []);
  const totalPages = Math.max(Math.ceil((data.total || 1) / state.limit), 1);
  pageInfoEl.textContent = `Page ${state.page + 1} / ${totalPages}`;
  document.getElementById('prev-page').disabled = state.page === 0;
  document.getElementById('next-page').disabled = state.page + 1 >= totalPages;
}

function renderItems(items) {
  itemsEl.innerHTML = '';
  if (!items.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No items found. Adjust filters or search keywords.';
    itemsEl.appendChild(empty);
    return;
  }

  items.forEach(item => {
    const card = document.createElement('article');
    card.className = 'item-card';

    const header = document.createElement('div');
    header.className = 'item-header';
    header.innerHTML = `
      <div>
        <h3>Article: ${item.title || 'Untitled Study'}</h3>
        <div class="meta">PMID ${item.pmid || 'n/a'} · ${item.journal || 'Journal n/a'} (${item.year || 'n/a'})</div>
      </div>
      ${state.dataset === 'qa' ? `<button class="secondary" data-compare="${item.pmid}" data-id="${item.id}">Compare QA options</button>` : ''}
    `;
    card.appendChild(header);

    if (state.dataset === 'qa') {
      card.appendChild(buildParagraph('qa-question', `Q: ${escapeHtml(item.question)}`));
      card.appendChild(buildParagraph('qa-answer', `<strong>Answer:</strong> ${escapeHtml(item.answer)}`));

      const rationale = document.createElement('details');
      rationale.className = 'rationale';
      rationale.innerHTML = `<summary>View rationale</summary><p>${escapeHtml(item.explanation)}</p>`;
      card.appendChild(rationale);
    } else {
      if (item.summary) {
        const summary = document.createElement('div');
        summary.className = 'extraction-summary';
        summary.innerHTML = `<strong>Summary:</strong> ${escapeHtml(item.summary)}`;
        card.appendChild(summary);
      }

      const grid = document.createElement('div');
      grid.className = 'section-stack';
      const fields = [
        { key: 'population', label: 'Population Studied' },
        { key: 'symptoms', label: 'Symptoms' },
        { key: 'riskFactors', label: 'Risk Factors / Triggers' },
        { key: 'interventions', label: 'Interventions / Treatments' },
        { key: 'outcomes', label: 'Outcomes' }
      ];
      fields.forEach(({ key, label }) => {
        const sec = document.createElement('div');
        sec.className = `section-card section-${key}`;
        const bullets = (item.categorized?.[key] || []).filter(Boolean);
        sec.innerHTML = `
          <h4>${label}</h4>
          ${bullets.length ? bullets.map(b => `<div>• ${escapeHtml(b)}</div>`).join('') : '<div class="empty">Not captured</div>'}
        `;
        grid.appendChild(sec);
      });
      card.appendChild(grid);

      const rawDetails = document.createElement('details');
      rawDetails.className = 'rationale';
      rawDetails.innerHTML = `
        <summary>Show raw extraction (with chain-of-thought)</summary>
        ${ (item.parsed?.sections || []).map(section => `<p><strong>${escapeHtml(section.heading)}</strong><br>${section.bullets.map(b => escapeHtml(b)).join('<br>')}</p>`).join('') }
      `;
      card.appendChild(rawDetails);
    }

    const abstract = document.createElement('div');
    abstract.className = 'abstract';
    abstract.innerHTML = `<strong>Abstract:</strong><br>${item.abstract}`;
    card.appendChild(abstract);

    const validation = document.createElement('div');
    validation.className = 'validation-block';

    evalQuestions[state.dataset].forEach(q => {
      const row = document.createElement('div');
      row.className = 'validation-question';
      row.innerHTML = `<p>${q.text}</p>`;
      const buttons = document.createElement('div');
      buttons.className = 'btn-group';

      const yesBtn = document.createElement('button');
      yesBtn.className = 'btn yes';
      yesBtn.textContent = 'Yes';
      yesBtn.onclick = () => openConfirm({ dataset: state.dataset, itemId: item.id, questionId: q.id, answer: 'yes', questionText: q.text });

      const noBtn = document.createElement('button');
      noBtn.className = 'btn no';
      noBtn.textContent = 'No';
      noBtn.onclick = () => openConfirm({ dataset: state.dataset, itemId: item.id, questionId: q.id, answer: 'no', questionText: q.text });

      buttons.appendChild(yesBtn);
      buttons.appendChild(noBtn);
      row.appendChild(buttons);
      validation.appendChild(row);
    });

    card.appendChild(validation);
    itemsEl.appendChild(card);
  });

  if (state.dataset === 'qa') {
    queryAll('[data-compare]').forEach(btn => {
      btn.addEventListener('click', () => openCompare(btn.dataset.compare));
    });
  }
}

function buildParagraph(className, html) {
  const p = document.createElement('p');
  p.className = className;
  p.innerHTML = html;
  return p;
}

function openConfirm(payload) {
  pendingResponse = payload;
  modalQuestion.textContent = payload.questionText;
  modalSubtext.textContent = `You selected “${payload.answer.toUpperCase()}”. Are you confident in this judgement?`;
  modal.classList.remove('hidden');
}

modalConfirm.onclick = async () => {
  if (!pendingResponse) return;
  await submitResponse({ ...pendingResponse, sure: true });
  closeModal();
};

modalCancel.onclick = async () => {
  if (!pendingResponse) return;
  await submitResponse({ ...pendingResponse, sure: false });
  closeModal();
};

function closeModal() {
  modal.classList.add('hidden');
  pendingResponse = null;
}

async function submitResponse(payload) {
  try {
    const res = await fetch('/api/responses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Failed to submit response');
    await refreshSummary();
    showToast('Response recorded');
  } catch (err) {
    console.error(err);
    showToast('Submission failed');
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 2200);
}

async function openCompare(pmid) {
  const params = new URLSearchParams({ dataset: state.dataset, pmid });
  const res = await fetch(`/api/compare-options?${params.toString()}`);
  const data = await res.json();
  const items = data.items || [];
  compareSelection = { pmid, choiceId: items[0]?.id || null };

  compareListEl.innerHTML = '';
  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'compare-card';
    card.innerHTML = `
      <header>
        <label>
          <input type="radio" name="compare-choice" value="${item.id}" ${item.id === compareSelection.choiceId ? 'checked' : ''} />
          ${item.question.slice(0, 80)}
        </label>
        <span class="meta">PMID ${item.pmid}</span>
      </header>
      <div><strong>Answer:</strong> ${item.answer}</div>
      <div><strong>Explanation:</strong> ${item.explanation}</div>
    `;
    queryAll('input[type="radio"]', card).forEach(radio => {
      radio.addEventListener('change', evt => {
        compareSelection.choiceId = evt.target.value;
      });
    });
    compareListEl.appendChild(card);
  });
  compareModal.classList.remove('hidden');
}

compareSubmit.onclick = async () => {
  if (!compareSelection || !compareSelection.choiceId) {
    showToast('Select an option first');
    return;
  }
  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: state.dataset, pmid: compareSelection.pmid, choiceId: compareSelection.choiceId })
    });
    if (!res.ok) throw new Error('Submit failed');
    showToast('Comparison saved');
    compareModal.classList.add('hidden');
  } catch (err) {
    console.error(err);
    showToast('Comparison failed');
  }
};

compareClose.onclick = () => {
  compareModal.classList.add('hidden');
};

summaryBtn.onclick = async () => {
  await refreshSummary();
  renderSummaryModal();
  summaryModal.classList.remove('hidden');
};

summaryClose.onclick = () => {
  summaryModal.classList.add('hidden');
};

function renderSummaryModal() {
  const summary = state.summary || { questionSummaries: {}, compare: {}, totalDecisions: 0 };
  summaryContent.innerHTML = '';
  exportButtons.innerHTML = '';

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.innerHTML = `<strong>Total decisions:</strong> ${summary.totalDecisions || 0}`;
  summaryContent.appendChild(meta);

  Object.entries(summary.questionSummaries || {}).forEach(([questionId, info]) => {
    const section = document.createElement('section');
    section.innerHTML = `<h4>${humanizeQuestion(questionId)}</h4>`;

    const countsRow = document.createElement('div');
    countsRow.className = 'badge-row';
    const counts = info.counts || {};
    ['yes_sure', 'yes_unsure', 'no_sure', 'no_unsure'].forEach(key => {
      if (counts[key]) {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = `${key.replace('_', ' ')} · ${counts[key]}`;
        countsRow.appendChild(badge);
      }
    });
    section.appendChild(countsRow);

    const yesList = buildRecordList(info.yes || [], 'Validated (Yes)');
    const noList = buildRecordList(info.no || [], 'Flagged (No)');
    section.appendChild(yesList);
    section.appendChild(noList);

    const exportRow = document.createElement('div');
    exportRow.className = 'export-group';
    exportRow.innerHTML = `
      <button class="secondary" data-export-question="${questionId}" data-decision="">Export CSV</button>
      <button class="secondary" data-export-question="${questionId}" data-decision="yes">Export Yes Only</button>
      <button class="secondary" data-export-question="${questionId}" data-decision="no">Export No Only</button>
    `;
    section.appendChild(exportRow);

    summaryContent.appendChild(section);
  });

  exportButtons.innerHTML = `
    <button class="secondary" data-export-all="csv">Download All (CSV)</button>
    <button class="secondary" data-export-all="json">Download All (JSON)</button>
  `;

  queryAll('[data-export-question]').forEach(btn => {
    btn.addEventListener('click', () => exportJudgements({
      questionId: btn.dataset.exportQuestion,
      decision: btn.dataset.decision || undefined,
      format: 'csv'
    }));
  });

  queryAll('[data-export-all]').forEach(btn => {
    btn.addEventListener('click', () => exportJudgements({ format: btn.dataset.exportAll }));
  });

}

function buildRecordList(records, heading) {
  const container = document.createElement('div');
  container.className = 'record-list';
  const title = document.createElement('h4');
  title.textContent = heading;
  container.appendChild(title);
  if (!records.length) {
    const p = document.createElement('p');
    p.textContent = 'No responses yet.';
    container.appendChild(p);
    return container;
  }
  records.slice(-5).reverse().forEach(record => {
    const item = document.createElement('div');
    item.className = 'record-item';
    item.innerHTML = `
      <h4>${escapeHtml(record.title || 'Untitled Study')} (${escapeHtml(record.pmid || 'PMID n/a')})</h4>
      <p><strong>Timestamp:</strong> ${escapeHtml(new Date(record.timestamp).toLocaleString())}</p>
      <p><strong>Details:</strong> ${escapeHtml(renderRecordSummary(record))}</p>
      <p><strong>Confidence:</strong> ${record.sure ? 'Reviewer confident' : 'Reviewer unsure'}</p>
    `;
    container.appendChild(item);
  });
  return container;
}

function renderRecordSummary(record) {
  if (record.dataset === 'qa') {
    return `${record.content.question} → ${record.content.answer}`;
  }
  return record.content.summary || 'See extraction notes';
}

function humanizeQuestion(id) {
  const question = [...evalQuestions.qa, ...evalQuestions.extracted].find(q => q.id === id);
  return question ? question.text : id;
}

async function exportJudgements({ questionId, decision, format = 'csv' }) {
  try {
    const params = new URLSearchParams({ dataset: state.dataset, format });
    if (questionId) params.append('questionId', questionId);
    if (decision) params.append('decision', decision);
    const res = await fetch(`/api/export?${params.toString()}`);
    if (!res.ok) throw new Error('Export failed');
    if (format === 'json') {
      const data = await res.json();
      download(JSON.stringify(data, null, 2), `${state.dataset}-${questionId || 'all'}.json`, 'application/json');
    } else {
      const blob = await res.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const csv = new TextDecoder().decode(arrayBuffer);
      download(csv, `${state.dataset}-${questionId || 'all'}${decision ? '-' + decision : ''}.csv`, 'text/csv');
    }
  } catch (err) {
    console.error(err);
    showToast('Export failed');
  }
}

function download(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.getElementById('prev-page').onclick = () => {
  if (state.page > 0) {
    state.page -= 1;
    loadItems();
  }
};

document.getElementById('next-page').onclick = () => {
  state.page += 1;
  loadItems();
};

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = searchInput.value.trim();
    state.page = 0;
    loadItems();
  }, 260);
});

function queryAll(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

init();
