"""Static evaluation page (WORKPLAN 2.7, spec section 10).

`build_report` renders report/index.html from saved files only: case sources, saved
target first answers, saved verifier/baseline verdicts and runs/metrics/metrics.json.
Missing data is shown as missing. Nothing is computed from the private key here
except the answer-bank labels, which are displayed as labels of constructed items.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from medproof.load_case import discover_cases, load_private, load_public, parse_source_lines, twin_dir
from medproof.runs import DEFAULT_ROOT, bank_item_id, load_target_runs, load_verifier_runs


def _chart(case_dir: Path) -> dict[str, Any]:
    pc = load_public(case_dir)
    return {
        "case_id": pc.case_id,
        "question": pc.question,
        "files": [
            {"name": name, "lines": [{"id": sl.line_id, "text": sl.text} for sl in parse_source_lines(name, text)]}
            for name, text in pc.sources.items()
        ],
    }


def _answers(case_dir: Path, targets_by_case: dict, verdicts: dict) -> list[dict[str, Any]]:
    priv = load_private(case_dir)
    out: list[dict[str, Any]] = []
    for run in targets_by_case.get(priv.case_id, []):
        out.append({
            "id": run.run_id,
            "kind": "target",
            "title": f"Target first answer ({run.model}, {run.timestamp_utc})",
            "text": run.answer_raw,
            "meta": {"stop_reason": run.stop_reason, "model": run.model, "timestamp_utc": run.timestamp_utc,
                     "input_tokens": run.input_tokens, "output_tokens": run.output_tokens},
            "verdicts": verdicts.get(run.run_id, {}),
        })
    for ans in priv.answer_bank:
        item_id = bank_item_id(priv.case_id, ans.answer_id)
        out.append({
            "id": item_id,
            "kind": "bank",
            "title": f"Constructed answer {ans.answer_id} (labeled {ans.label}; not a target output)",
            "text": ans.text,
            "meta": {"label": ans.label, "trap": ans.trap.value if ans.trap else None, "style": ans.style},
            "verdicts": verdicts.get(item_id, {}),
        })
    return out


def build_report_data(cases_root: Path, runs_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    runs_root = Path(runs_root)
    targets_by_case: dict[str, list] = {}
    for run in load_target_runs(runs_root):
        targets_by_case.setdefault(run.case_id, []).append(run)
    verdicts: dict[str, dict[str, Any]] = {}
    for vr in load_verifier_runs(runs_root):
        verdicts.setdefault(vr.target_run_id, {})[vr.verifier_variant] = vr.model_dump(mode="json")
    cases = []
    for d in discover_cases(cases_root):
        entry = {"original": {**_chart(d), "answers": _answers(d, targets_by_case, verdicts)}, "twin": None}
        if (t := twin_dir(d)) is not None:
            entry["twin"] = {**_chart(t), "answers": _answers(t, targets_by_case, verdicts)}
        cases.append(entry)
    metrics_path = runs_root / "metrics" / "metrics.json"
    metrics: Optional[dict] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    variants = sorted({v for per in verdicts.values() for v in per})
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": cases,
        "variants": variants,
        "metrics": metrics,
    }


def build_report(cases_root: Path, runs_root: Path = DEFAULT_ROOT, output: Path = Path("report/index.html")) -> Path:
    data = build_report_data(cases_root, runs_root)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = Template(TEMPLATE).render(data_json=blob)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedProof Report</title>
<style>
:root{--bg:#f7f8f8;--panel:#ffffff;--ink:#22272b;--muted:#5d666d;--line:#dde2e4;--teal:#2f7f7a;
--teal-soft:#e3f1ef;--red:#b3261e;--red-soft:#fbe9e7;--amber:#8a5a00;--amber-soft:#fdf3dc;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;color:var(--muted)}
.banner{background:var(--amber-soft);color:var(--amber);padding:6px 24px;font-size:13px;border-bottom:1px solid var(--line)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 24px;background:var(--panel);border-bottom:1px solid var(--line)}
.controls label{font-size:13px;color:var(--muted)}
select,button{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}
button.toggle[aria-pressed=true]{background:var(--teal);color:#fff;border-color:var(--teal)}
.question{padding:12px 24px;font-weight:600;background:var(--panel);border-bottom:1px solid var(--line)}
.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr) minmax(0,1fr);gap:16px;padding:16px 24px}
@media (max-width:1000px){.grid{grid-template-columns:1fr;padding:16px}}
.col{background:var(--panel);border:1px solid var(--line);border-radius:10px;min-width:0;display:flex;flex-direction:column}
.col h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
.tabs{display:flex;flex-wrap:wrap;gap:4px;padding:8px 10px;border-bottom:1px solid var(--line)}
.tabs button{font-size:13px}
.tabs button[aria-selected=true]{background:var(--teal-soft);border-color:var(--teal);color:var(--teal)}
.chart{padding:6px 0;max-height:70vh;overflow:auto;font-family:var(--mono);font-size:12.5px}
.ln{display:grid;grid-template-columns:4.2em 1fr;gap:6px;padding:2px 12px;border-left:3px solid transparent}
.ln .id{color:var(--muted)}
.ln.hl{background:var(--teal-soft);border-left-color:var(--teal)}
.ln.hl-bad{background:var(--red-soft);border-left-color:var(--red)}
.body{padding:12px 14px;overflow:auto;max-height:70vh}
.answer{white-space:pre-wrap;margin:0;font-size:14px}
.meta{font-size:12px;color:var(--muted);margin-bottom:8px}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid var(--line);margin-right:6px}
.pill.kind-target{background:var(--teal-soft);color:var(--teal);border-color:var(--teal)}
.pill.kind-bank{background:#eef0f1;color:var(--muted)}
.overall{font-weight:700;padding:8px 10px;border-radius:8px;margin-bottom:10px}
.o-ERROR_DETECTED{background:var(--red-soft);color:var(--red)}
.o-NO_ISSUE_FOUND{background:var(--teal-soft);color:var(--teal)}
.o-INSUFFICIENT_TO_VERIFY,.o-TARGET_REFUSAL{background:var(--amber-soft);color:var(--amber)}
.card{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px;cursor:pointer}
.card:hover,.card:focus{border-color:var(--teal);outline:none}
.card .st{font-size:12px;font-weight:700}
.s-CONTRADICTED{color:var(--red)} .s-SUPPORTED{color:var(--teal)} .s-NOT_ESTABLISHED,.s-CONFLICTED{color:var(--amber)}
.card .cites{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:4px}
.empty{color:var(--muted);font-style:italic}
section.exp{padding:0 24px 32px}
section.exp h2{font-size:18px}
table{border-collapse:collapse;width:100%;background:var(--panel);font-size:13px;margin-bottom:18px}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f0f2f3}
.tablewrap{overflow-x:auto}
.note{color:var(--muted);font-size:13px}
</style>
</head>
<body>
<div class="banner">Research prototype. All records are SYNTHETIC; not a medical device and not clinical decision support.</div>
<header>
  <h1>MedProof</h1>
  <p>A medical AI answer can be plausible and still contradict the record.</p>
  <p class="note" id="generated"></p>
</header>
<div class="controls">
  <label for="caseSel">Case</label><select id="caseSel"></select>
  <button class="toggle" id="twinBtn" aria-pressed="false" type="button">Counterfactual twin: off</button>
  <label for="ansSel">Answer</label><select id="ansSel"></select>
  <label for="varSel">Verifier variant</label><select id="varSel"></select>
</div>
<div class="question" id="question"></div>
<div class="grid">
  <div class="col"><h2>Original chart</h2><div class="tabs" id="tabs" role="tablist"></div><div class="chart" id="chart"></div></div>
  <div class="col"><h2>Answer under review</h2><div class="body"><div class="meta" id="ansMeta"></div><pre class="answer" id="answer"></pre></div></div>
  <div class="col"><h2>Verified findings</h2><div class="body" id="findings"></div></div>
</div>
<section class="exp" id="exp"></section>
<script id="data" type="application/json">{{ data_json }}</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = String(text); return e; };
let state = {c: 0, twin: false, ans: 0, variant: null, file: 0, hl: {}};
$('generated').textContent = 'Generated ' + D.generated_at_utc + ' from saved runs.';

function current(){ const c = D.cases[state.c]; return state.twin && c.twin ? c.twin : c.original; }

function renderCaseSel(){
  const s = $('caseSel'); s.innerHTML = '';
  D.cases.forEach((c, i) => { const o = el('option', null, c.original.case_id); o.value = i; s.appendChild(o); });
  if (!D.cases.length) { $('question').textContent = 'No cases found.'; }
}
function renderAnsSel(){
  const s = $('ansSel'); s.innerHTML = '';
  const v = current();
  v.answers.forEach((a, i) => { const o = el('option', null, a.title); o.value = i; s.appendChild(o); });
  if (!v.answers.length) { const o = el('option', null, 'No answers saved for this case'); s.appendChild(o); }
  state.ans = Math.min(state.ans, Math.max(0, v.answers.length - 1)); s.value = state.ans;
}
function renderVarSel(){
  const s = $('varSel'); s.innerHTML = '';
  const vars = D.variants.length ? D.variants : ['(none saved)'];
  vars.forEach(v => { const o = el('option', null, v); o.value = v; s.appendChild(o); });
  if (!state.variant || !vars.includes(state.variant)) state.variant = vars.includes('haiku') ? 'haiku' : vars[0];
  s.value = state.variant;
}
function renderChart(){
  const v = current(); const tabs = $('tabs'); tabs.innerHTML = '';
  $('question').textContent = (state.twin ? '[TWIN] ' : '') + v.case_id + ': ' + v.question;
  v.files.forEach((f, i) => {
    const b = el('button', null, f.name); b.type = 'button'; b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', String(i === state.file));
    b.onclick = () => { state.file = i; renderChart(); };
    tabs.appendChild(b);
  });
  const chart = $('chart'); chart.innerHTML = '';
  const f = v.files[state.file] || v.files[0];
  if (!f) return;
  f.lines.forEach(l => {
    const row = el('div', 'ln'); row.id = 'ln-' + f.name + '-' + l.id;
    const key = f.name + '|' + l.id;
    if (state.hl[key] === 'bad') row.classList.add('hl-bad'); else if (state.hl[key]) row.classList.add('hl');
    row.appendChild(el('span', 'id', l.id)); row.appendChild(el('span', null, l.text));
    chart.appendChild(row);
  });
}
function renderAnswer(){
  const a = current().answers[state.ans];
  const meta = $('ansMeta'); meta.innerHTML = '';
  if (!a) { $('answer').textContent = ''; meta.appendChild(el('span', 'empty', 'No target first answer or bank item saved for this case yet.')); return; }
  meta.appendChild(el('span', 'pill kind-' + a.kind, a.kind === 'target' ? 'Real target first answer' : 'Constructed answer bank'));
  const bits = Object.entries(a.meta).filter(([k, v]) => v !== null && v !== undefined).map(([k, v]) => k + ': ' + v);
  meta.appendChild(el('div', null, bits.join(' | ')));
  $('answer').textContent = a.text || '(empty answer)';
}
function renderFindings(){
  const box = $('findings'); box.innerHTML = '';
  const a = current().answers[state.ans];
  const vr = a && a.verdicts ? a.verdicts[state.variant] : null;
  if (!vr) { box.appendChild(el('p', 'empty', 'No saved verdict for this answer and variant.')); return; }
  box.appendChild(el('div', 'overall o-' + vr.overall, vr.overall.replace(/_/g, ' ')));
  if (vr.explanation) box.appendChild(el('p', null, vr.explanation));
  if (vr.traps_found && vr.traps_found.length) box.appendChild(el('p', 'note', 'Reasoning traps flagged: ' + vr.traps_found.join(', ')));
  (vr.needs_review_reasons || []).forEach(r => box.appendChild(el('p', 'note', 'Needs review: ' + r)));
  if (!vr.claim_verdicts.length) box.appendChild(el('p', 'empty', 'No claim-level findings in this verdict.'));
  vr.claim_verdicts.forEach(cv => {
    const card = el('div', 'card'); card.tabIndex = 0;
    card.appendChild(el('div', 'st s-' + cv.status, cv.status.replace(/_/g, ' ') + ' | ' + cv.severity));
    card.appendChild(el('div', null, cv.claim_text));
    if (cv.why) card.appendChild(el('div', 'note', cv.why));
    if (cv.uncertainty) card.appendChild(el('div', 'note', 'Unknown: ' + cv.uncertainty));
    if (cv.citations.length) card.appendChild(el('div', 'cites', cv.citations.map(c => c.file + ':' + c.line_id + (c.quote ? ' "' + c.quote + '"' : '')).join('  ')));
    const go = () => {
      state.hl = {};
      const bad = cv.status === 'CONTRADICTED';
      cv.citations.forEach(c => { state.hl[c.file + '|' + c.line_id] = bad ? 'bad' : 'ok'; });
      const first = cv.citations[0];
      if (first) { const idx = current().files.findIndex(f => f.name === first.file); if (idx >= 0) state.file = idx; }
      renderChart();
      if (first) { const row = document.getElementById('ln-' + first.file + '-' + first.line_id); if (row) row.scrollIntoView({block: 'center'}); }
    };
    card.onclick = go; card.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
    box.appendChild(card);
  });
  const cost = el('p', 'note', 'Model calls: ' + vr.total_model_calls + ' | tokens in/out: ' + vr.input_tokens + '/' + vr.output_tokens +
    ' | estimated cost: $' + Number(vr.estimated_cost_usd).toFixed(6) + ' | latency: ' + vr.latency_ms + ' ms');
  box.appendChild(cost);
}
function fmtRate(b){ return b && b.n ? (b.k + '/' + b.n + ' (' + Math.round(100 * b.rate) + '%)') : 'n/a'; }
function fmt(x, d){ return (x === null || x === undefined) ? 'n/a' : Number(x).toFixed(d); }
function renderExperiment(){
  const box = $('exp'); box.innerHTML = '';
  box.appendChild(el('h2', null, 'Experiment'));
  if (!D.metrics) { box.appendChild(el('p', 'empty', 'No metrics yet. Run: python -m medproof.cli evaluate')); return; }
  (D.metrics.notes || []).forEach(n => box.appendChild(el('p', 'note', n)));
  ['target', 'bank'].forEach(src => {
    const t = D.metrics.tables[src]; if (!t) return;
    box.appendChild(el('h3', null, (src === 'target' ? 'Real target first answers' : 'Constructed answer bank') + ' (' + t.n_items + ' items)'));
    box.appendChild(el('p', 'note', t.description));
    const vs = Object.entries(t.variants);
    if (!vs.length) { box.appendChild(el('p', 'empty', 'No verdicts saved yet.')); return; }
    const wrap = el('div', 'tablewrap'); const tb = el('table');
    const head = ['Variant', 'Catch rate (wrong flagged)', 'False-alarm rate (correct flagged)', 'Insufficient on wrong', 'False abstentions', 'Citation validity', 'Mean cost / verdict (USD)', 'Median latency (ms)', 'p90 latency (ms)', 'Missing', 'Unlabeled', 'Refusals'];
    const hr = el('tr'); head.forEach(h => hr.appendChild(el('th', null, h))); tb.appendChild(hr);
    vs.forEach(([v, m]) => {
      const r = el('tr');
      const ev = m.evidence_validity;
      [v + (m.stub_verdicts ? ' (stub verifier)' : ''), fmtRate(m.error_recall), fmtRate(m.false_alarm_rate), m.abstention.insufficient_on_wrong,
       m.abstention.false_abstentions_on_correct, ev.valid + '/' + ev.citations, fmt(m.cost_usd.mean, 6), fmt(m.latency_ms.median, 0),
       fmt(m.latency_ms.p90, 0), m.missing_verdicts, m.unlabeled, m.target_refusals].forEach(x => r.appendChild(el('td', null, x)));
      tb.appendChild(r);
    });
    wrap.appendChild(tb); box.appendChild(wrap);
  });
  const tr = D.metrics.target_runs || [];
  box.appendChild(el('h3', null, 'Target first answers'));
  if (!tr.length) { box.appendChild(el('p', 'empty', 'No official target runs saved yet.')); return; }
  const wrap = el('div', 'tablewrap'); const tb = el('table'); const hr = el('tr');
  ['Run', 'Case', 'Twin', 'Model', 'Stop reason', 'Human label', 'Trap'].forEach(h => hr.appendChild(el('th', null, h))); tb.appendChild(hr);
  tr.forEach(x => { const r = el('tr'); [x.run_id, x.case_id, x.is_twin ? 'yes' : 'no', x.model, x.stop_reason, x.label || 'unlabeled', x.trap || ''].forEach(v => r.appendChild(el('td', null, v))); tb.appendChild(r); });
  wrap.appendChild(tb); box.appendChild(wrap);
}
function renderAll(){
  const c = D.cases[state.c];
  const btn = $('twinBtn');
  btn.disabled = !(c && c.twin); if (!(c && c.twin)) state.twin = false;
  btn.setAttribute('aria-pressed', String(state.twin)); btn.textContent = 'Counterfactual twin: ' + (state.twin ? 'on' : 'off');
  state.hl = {}; state.file = 0;
  renderAnsSel(); renderVarSel(); renderChart(); renderAnswer(); renderFindings();
}
$('caseSel').onchange = e => { state.c = +e.target.value; state.ans = 0; state.twin = false; renderAll(); };
$('twinBtn').onclick = () => { state.twin = !state.twin; state.ans = 0; renderAll(); };
$('ansSel').onchange = e => { state.ans = +e.target.value; state.hl = {}; renderChart(); renderAnswer(); renderFindings(); };
$('varSel').onchange = e => { state.variant = e.target.value; state.hl = {}; renderChart(); renderFindings(); };
renderCaseSel(); renderExperiment();
if (D.cases.length) {
  // Deep link: #case=med_01&twin=1&answer=<answer id>&variant=haiku&finding=1
  const h = new URLSearchParams(location.hash.slice(1));
  const ci = D.cases.findIndex(c => c.original.case_id === h.get('case'));
  if (ci >= 0) { state.c = ci; $('caseSel').value = ci; }
  if (h.get('twin') === '1' && D.cases[state.c].twin) state.twin = true;
  if (h.get('variant')) state.variant = h.get('variant');
  renderAll();
  const ai = current().answers.findIndex(a => a.id === h.get('answer'));
  if (ai >= 0) { state.ans = ai; $('ansSel').value = ai; renderChart(); renderAnswer(); renderFindings(); }
  const fi = parseInt(h.get('finding') || '0', 10);
  if (fi > 0) { const cards = document.querySelectorAll('#findings .card'); if (cards[fi - 1]) cards[fi - 1].click(); }
}
})();
</script>
</body>
</html>
"""
