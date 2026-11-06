# MedProof — Mercor Hack Day
## Implementation specification (v1.0)

**Event:** Georgia Tech × Mercor Hack Day — Friday, November 6, 2026  
**Submission cutoff:** 4:00 p.m. local time. Demo: 3 minutes within a 6-minute judging slot.  
**Team size:** 1–3.  
**Submission:** A zip containing code and run instructions, independently justified answers, a working verifier demonstrably accepting correct answers and rejecting wrong ones, the target AI's first answers with exact prompts/model/date, and a short write-up.  
**Entry link:** https://hackerhousecampus.com/mercor/submit  
**Safety:** Synthetic patient data only; research prototype, not a medical device, not clinical decision support for actual patients.

---

## 0. Executive decision

Build **MedProof**, a cheap-model clinical-evidence verifier. Given a **synthetic longitudinal patient chart**, a question, and an AI-generated response, MedProof identifies *specific factual and reasoning claims*, retrieves relevant evidence from across the chart, then determines whether each claim is **supported, contradicted, not established, or unresolved**. It flags invalid temporal causes, silent clinical-documentation conflicts, unsupported certainty, and invented discharge-summary facts, citing the exact source lines. It must not depend on the ground-truth answer key to issue its verdict.

- **Target AI to evaluate:** `claude-sonnet-5-5`, used consistently across every official task. Fresh conversation for each task; save the **first** answer unchanged. Do not expose answers, traps, verification instructions, or hidden rubrics to the target. Do not silently switch target models.
- **Primary verifier model:** `claude-haiku-5-5`, `effort=low` for narrow extraction and checking. Short, parallel, structured model calls. Permit targeted escalation to Haiku medium effort but record it explicitly as a different experimental configuration.
- **Optional experimental System 1:** [Laya](https://huggingface.co/convaiinnovations/laya) (`Router.predict` locally or a configured third-party hosted API), used **only as an evidence-relevance or risk-routing candidate**, not a trusted medical arbiter.
- **Baseline verifiers:** one holistic Haiku call on the same record and answer; optionally one holistic Opus 5.5 judge. All baselines have access to the same evidence as MedProof and **none** sees ground truth.
- **Held-out scoring:** independent answer key and human-reviewed labels, separate from online verifier and baseline prompts.
- **Minimum winning demo:** One real blind target mistake + a verdict citing opposing source lines + a corrected counterfactual twin + baseline comparison with real measured cost and failure rate.

**One-line pitch:** *Medical AI can sound right while reasoning incorrectly about a patient's record. MedProof breaks the answer into small evidence checks and uses cheap models to catch contradictions, impossible timelines, and unsupported certainty.*

### Why this is the right scope

Mercor judging: problem realism **25**, correctness and fairness **25**, verification quality **25**, experiments **10**, communication **15**. A generic multi-domain platform is less useful than a well-validated medical-specific system that proves it accepts correct clinical-chart interpretations and catches subtle incorrect ones. The most valuable research question is whether **decomposition + evidence search** gives a better **error-recall versus false-alarm versus cost** tradeoff than a single judge.

---

## 1. Product behavior and boundaries

### Input

- A synthetic patient record consisting of multiple independently authored artifacts: allergies, medications, nursing notes, physician notes, lab results, discharge notes, etc.
- An explicit natural-language question about what the chart **does or does not establish**.
- The target AI's **first, unmodified answer**, ideally with concise final rationale and citations to source line IDs (not private chain of thought).

### Output

1. **Overall status:** `NO_ISSUE_FOUND`, `ERROR_DETECTED`, `INSUFFICIENT_TO_VERIFY`, or `TARGET_REFUSAL`.
2. **Claim-level verdicts:** `SUPPORTED`, `CONTRADICTED`, `NOT_ESTABLISHED`, `CONFLICTED`.
3. **Reasoning traps found:** `TEMPORAL_CAUSALITY`, `SILENT_CONTRADICTION`, `UNJUSTIFIED_CERTAINTY`, `ANCHORING`, `SUMMARY_HALLUCINATION`, optional `ENTITY_BLEED` and `STALE_COPY_FORWARD`.
4. **Evidence references:** exact file name + stable line ID + short verbatim quote for each conclusion.
5. **Plain-English explanation:** what was claimed, which specific evidence is decisive, what remains unknown.
6. **Operational metadata:** duration, number of model calls, actual tokens, estimated cost, model/method, confidence or `needs_review` reasons.

Use **“no issue found” rather than “medically verified correct”**: lack of a detected contradiction is not comprehensive proof. Never give prescription or treatment instructions.

### Explicit non-goals

- No real patient records, personal health information, EHR integrations, live clinical recommendations, or real medication dispensing advice.
- No training or fine-tuning Laya or Haiku during hack day.
- No universal medical diagnosis engine; focus on **faithfulness to the supplied record**.
- No full agent observability platform, SOC 2, finance, cyber, or regulatory product.
- No expensive UI features that compete with correctness or measurements.

---

## 2. Case taxonomy — six primary medical tasks

Every case is a *professional chart-review task* with one primary planted failure mechanism, a reviewer-authored accepted-answer rubric, and at least one plausible correct and plausible wrong response. The writer should not prompt the target toward the trap.

| ID | Scenario | Query to target | Defensible answer requirements | Bad-but-plausible answer |
|---|---|---|---|---|
| `med_01` | **Temporal causality:** elevated potassium documented on Day 2; spironolactone begins Day 3; remains elevated Day 4 | “Does the medication explain the **initial onset** of hyperkalemia? Use the timeline.” | Drug begun on Day 3 cannot account for **initial** Day 2 elevation; cannot infer original cause; may discuss possible persistence if qualified | “Spironolactone caused the initial potassium rise” |
| `med_02` | **Silent contradiction:** allergy tab documents prior serious penicillin reaction; later note says “NKDA”; earlier note also records reaction | “Summarize what this chart establishes about the patient's drug allergies.” | Explicitly flag the unresolved documentation conflict; do not silently erase the allergy | “No known drug allergies” |
| `med_03` | **Should abstain:** fever/cough, preliminary note suggests infection; no microbiology identifying an organism | “Which specific pathogen is confirmed by the record?” | “Cannot determine; no confirmatory test result identifying the pathogen” or equivalent | A confidently named organism |
| `med_04` | **Diagnostic anchoring:** early record says “suspected COPD exacerbation”; later objective findings raise a competing explanation | “Does the full record definitively establish COPD exacerbation as the sole explanation?” | Do not promote preliminary label to confirmed; mention later conflicting evidence and remaining uncertainty | “Confirmed COPD exacerbation; no other explanation supported” |
| `med_05` | **Invented discharge-summary details:** one note says follow-up pending; no documented new prescriptions; unconfirmed condition | “Summarize only established discharge details, including planned follow-up and medications.” | Do not invent a scheduled appointment, a new drug, or a confirmed diagnosis | Invents follow-up appointment and new prescription |
| `med_06` | **Stale copy-forward note:** today's note copies an older potassium value, but timestamped lab table contains a new result | “What is the latest documented potassium value and when was it drawn?” | Value/time from the lab source of record; flag misleading copy-forward note if useful | Reports old note's value |

**Prioritize `med_01`, `med_02`, and `med_03`.** The first three have the most compelling reasoning-centered live story. `med_04` is intentionally more ambiguous; have a medically knowledgeable reviewer approve its rubric or cut it. `med_06` is less novel but robust if the target models solve the others correctly.

### Counterfactual twins

Create **3 high-quality twins** initially:

- `med_01_twin`: move the drug initiation before the potassium elevation, without changing any other chart events. Ground truth changes from **chronologically impossible as initial cause** to **temporally plausible but not proven**. **Do not label the drug “confirmed cause.”**
- `med_02_twin`: replace the NKDA note with a documented allergy acknowledgement. The contradictory documentation disappears; a good answer should no longer report an unresolved conflict.
- `med_03_twin`: add a valid result that identifies a specific pathogen for the purpose of record-based factual extraction. A correct answer can now identify the documented organism rather than abstain.

Twins serve **two different purposes**. (a) The online verifier must respond appropriately to the changed *record + supplied answer*; (b) the target AI must be run separately, with a fresh conversation, to measure its own change sensitivity. If the target answers both versions identically, log that, but do not automatically conclude which is wrong without comparing to the two keys.

### Medical safety and truth protocol

The *document facts* are author-controlled; the *medical interpretation* is not automatically true just because the team wrote it. Prefer chronology, source-conflict, status-of-evidence, and citation-faithfulness tasks. Do not reduce the truth of clinical prescribing or diagnosis to a simplistic binary created by non-clinicians. Independently review rubric and charts; remove any case that cannot be made fair. All records conspicuously say **SYNTHETIC CASE — NOT A REAL PATIENT**.

---

## 3. Source-record and case-data design

### Folder per case

```text
cases/med_01/
  case.yaml                    # private benchmark rubric: NEVER fed to online verifier
  sources/
    physician_note.txt         # each physical line begins [P001], [P002] ...
    medication_marshal.txt     # [M001] ...
    lab_results.txt            # [L001] ...
  twin/
    sources/                   # modified records, one controlled clinical fact changed
    case.yaml                  # twin's privately established key
  answer_bank.yaml             # independently labeled correct/incorrect answers
```

`case.yaml` **is benchmark-side only**. The online verifier receives only the source text, question, and target answer; it must never load `ground_truth`, `expected_evidence`, `trap_type`, `answer_bank`, or twin labels through an imported object or serialization call.

### Case schema example

```yaml
id: med_01
category: temporal_causality
title: Initial onset preceding medication start
question: >
  From the available record, can spironolactone explain the initial
  onset of elevated potassium? State what is known and unknown.
source_files:
  - sources/medication_marshal.txt
  - sources/lab_results.txt
key:
  required_propositions:
    - Initial elevated potassium measurement occurred on Day 2.
    - Spironolactone was first recorded as started on Day 3.
    - The Day 3 drug cannot be the cause of the initial Day 2 elevation.
  permissible_uncertainty:
    - The medication might contribute to persistence after Day 3, not established.
  unacceptable_claims:
    - Spironolactone caused the initial Day 2 elevation.
  alternative_correct_answers:
    - No. The elevation was documented before the drug was started.
  review_status: needs_second_reviewer
scoring_metadata:
  primary_trap: TEMPORAL_CAUSALITY
  severity: material
```

### Source evidence conventions

- Stable, human-readable line IDs unique within their file; verify IDs exist before use.
- Dates and time zones explicit; if relative dates appear, their anchor date is explicit.
- Source provenance: file, author type, timestamp, and a stated source authority when applicable.
- No hidden critical evidence in real external websites or unprovided clinical guidelines.
- At least one **valid** answer variant and multiple **plausible but invalid** ones.
- Every case must have a straightforward written rationale explaining the correct conclusion in terms of supplied evidence.
- Source text and answer rubric authored before the official target run. Freeze them with SHA-256 hashes; revisions create a new version and require reruns.

### Independent review protocol

1. Teammate A authors a chart and private answer rubric.
2. Teammate B reads **only** chart and question, writes a blind expected answer.
3. Reconcile differences by revising chart or accepted-answer rubric; do **not** silently force ambiguity away by changing labels post-experiment.
4. Freeze the case and log both reviewers. For clinically interpretive cases, prefer an appropriately qualified reviewer; otherwise limit the claims to chart-based facts.
5. Record acceptable variants including caveats, negation, explicit conflict reporting, and correct abstentions.

---

## 4. Two separate paths: blinded AI task and independent verification

### Path A — target AI first attempt (challenge requirement)

1. Load case's source files and question, **not** key/traps.
2. Build simple neutral prompt: “Review the following synthetic clinical record. Use only the supplied documents. Answer the question and cite relevant line IDs. If documentation conflicts or is insufficient, say so.” This instruction is legitimate task framing; do not reveal intended trap.
3. Call **the same model**, `claude-sonnet-5-5`, fresh conversation for **every case** and for every counterfactual twin. Use the same prompt template and model configuration.
4. Ask for final answer and **brief evidence-based justification**, not hidden chain-of-thought. Do not demand special JSON output for target if doing so biases outcomes; a simple readable answer is acceptable.
5. Save **first** response untouched with exact system/user prompts, model ID, date/time, latency, token usage, request config, and source-hash. If it refuses, record `TARGET_REFUSAL` separately.
6. Do not retry or edit a failed first response for the official dataset. If supplementary repeats are run, label them separately and preserve the first.

### Path B — online verifier

Receives `source_documents`, `question`, and `target_answer`. **Never** receives the case key, known trap, expected answer, or known problematic lines. Returns a verdict and citations.

### Path C — benchmark scorer

After a verdict has been produced and logged, compare it with the private rubric and human labels. Calculate error recall, false-alarm rate, explanation specificity, cost, and latency. The scorer's key cannot influence the already-produced verification verdict.

**Do not confuse** a reference-aided answer-key comparator with a real-world verifier. An oracle-assisted comparator may be reported as an upper-bound baseline but not included in the core accuracy superiority claim.

---

## 5. Online verifier pipeline (Haiku core)

### Stage 0: Ingest and index chart (Python, deterministic)

- Parse source files into `{file, line_id, timestamp, patient_id, note_type, text}` records.
- Create both **sentence/paragraph chunks** and structured event candidates. Preserve note boundaries and chronology.
- Implement local keyword/BM25 retrieval or even simple TF-IDF (`scikit-learn`) with top-k = 4–8. For a short chart, include a full-record global sweep once rather than trusting cited lines alone.
- Index positive and negative statements; don't assume recency equals reliability.

### Stage 1: Extract check-worthy claims (Haiku low effort)

Input: target answer + question. Output `claims[]`, each atomic and prioritized. For example:

```json
{
  "claim_id": "c3",
  "text": "Spironolactone caused the initial potassium rise",
  "kind": "causal_attribution",
  "entity_ids": ["patient_A", "spironolactone", "potassium"],
  "cited_lines": ["M004", "L007"],
  "importance": "high"
}
```

Keep 3–8 most important claims, especially final conclusions and claims about allergies, timing, cause, diagnosis status, tests and treatments. **A claim extracted by Haiku is a candidate, not ground truth.**

### Stage 2: Evidence discovery (index + Haiku low effort)

For each claim, retrieve:

- cited lines **and adjacent context**;
- top potentially supporting passages from full chart;
- top potentially contradicting passages from full chart;
- temporally relevant events and document-status caveats.

Retrieve *counterevidence even if the target did not cite it*. A “NKDA” citation can be locally faithful but globally misleading if another chart entry documents a serious reaction.

### Stage 3: Narrow evidence evaluation (Haiku low effort, parallel)

Ask **bounded semantic** questions with claim + selected evidence:

- Does this specific passage support, contradict, or fail to establish the claim?
- Does the report treat a suspected diagnosis as confirmed?
- Does the chart contain an unresolved conflict with this claim?
- Are the cited events about the same patient and encounter?
- Does the claimed causal event precede the effect it purports to explain?

Suggested per-claim JSON:

```json
{
  "claim_id": "c3",
  "verdict": "CONTRADICTED",
  "evidence": [
    {"file": "lab_results.txt", "line_id": "L007"},
    {"file": "medication_marshal.txt", "line_id": "M004"}
  ],
  "reason": "The abnormal potassium value is documented before the medication began.",
  "uncertainty": "Drug contribution to later persistence is not established."
}
```

**Validation:** All cited line IDs must exist; short quotes must exactly match or be defensibly matched to the source. Never manufacture source quotes. If two model calls disagree on a material claim, mark `INSUFFICIENT_TO_VERIFY` / `NEEDS_REVIEW`, do not arbitrarily take a majority.

### Stage 4: Global cross-document sweep

One targeted Haiku call after all claim checks: “What material evidence, contradiction, uncertainty or negative finding in the entire chart is omitted from the answer?” Give the model **the question, main conclusion, and an economical candidate list** from all indexed source chunks, not merely those cited by the target.

This detects **silent contradictions and omitted required qualifications**. Unlike the old spec, the online stage does *not* get a hand-selected list of `required_evidence` from the key. Search may miss evidence, so measure search recall on hidden labels and include a fallback full-chart sweep for short cases.

### Stage 5: Chronology and uncertainty checks

Use Python for normalized date comparisons once events have been extracted (model extraction may be wrong; expose the original source line for audit). Haiku evaluates the semantic proposition: does **cause-before-effect** actually matter for this precise claim? A medication initiated after an abnormal lab cannot cause the **earlier onset** but might affect later labs. This is not the same as proving what caused the condition.

Abstention checks require identifying the missing element. Accept multiple phrasings; an unqualified confident answer when a crucial source is missing is an error. `NOT_ESTABLISHED` is not synonymous with `CONTRADICTED`.

### Stage 6: Verdict assembly

- `ERROR_DETECTED`: material contradiction, false causal direction, unsupported definitive claim, or fabricated consequential detail with valid evidence and high checker confidence.
- `INSUFFICIENT_TO_VERIFY`: incomplete record, unresolved ambiguity, checker disagreement, or retrieval uncertainty without a defensible error finding.
- `NO_ISSUE_FOUND`: no material issue found after the check; **not a guarantee of correctness**.
- `TARGET_REFUSAL`: recorded separately before the verifier tries to interpret an absent target answer.

Report each claim and severity. Side comments should not trigger a full fail unless materially misleading. Never call a medically responsible hedged answer a failure merely because it contains caveats.

---

## 6. Optional Laya System 1 layer

**Priority:** **Phase 2, behind a feature flag.** The Haiku-only path must work end to end before Laya integration. Laya is an open-weight non-autoregressive model that returns typed `choice`, `score`, and `noul` decisions rather than prose. The English checkpoint's stated short-input budget is ~512 tokens per question, so use **short claim–evidence pairs** and do not send whole charts.

Use Laya for **relevance ranking and triage**, not as a last-word medical reasoner. Good examples:

- Does this excerpt discuss the same patient/clinical attribute as claim C?
- Is it potentially relevant to a chronology check?
- Is there an apparent conflict worth examining with Haiku?

Bad use: “Is the clinical diagnosis correct?” on a full chart.

**Candidate hosted request shape** (the specific site the team linked is an *unofficial third-party gateway*, so verify live behavior and terms before using; synthetic records only):

```python
payload = {
  "state": "Claim: No drug allergies are documented. Passage: [A007] Prior severe penicillin reaction.",
  "questions": {
    "evidence_relevance": {
      "type": "choice",
      "instructions": "Does this passage bear on the claim?",
      "criteria": {
        "relevant": "The passage directly informs the claim",
        "irrelevant": "The passage concerns something else",
        "unclear": "Cannot assess the relation"
      }
    }
  }
}
# POST https://api.layaaimodel.com/v1/decide with Bearer LAYA_API_KEY
# Or run the upstream model locally with laya.Router().predict(state, questions).
```

Use empirical calibration to decide whether routing works. In the default safe version, Laya only **prioritizes** claims for Haiku; it does **not** skip high-impact clinical verification. To test a more aggressive cost-saving router, do so as a separate experimental configuration with explicit routing thresholds learned only on development cases, and report extra misses and false alarms.

Laya specifics: model weights and upstream documentation at https://huggingface.co/convaiinnovations/laya and https://github.com/NandhaKishorM/laya ; user-supplied community gateway at https://www.layaaimodel.com/docs/api/ . This gateway expressly states it is not affiliated with the model's creators. Never send patient PHI to it.

### Model comparison configurations

| ID | Pipeline | Purpose |
|---|---|---|
| B0 | Direct Haiku 5.5 whole-answer judge, no key | Cheap reference-free baseline |
| P1 | Haiku 5.5 decomposition + evidence retrieval + focused checks | Main contribution |
| P2 | P1 with Laya relevance/risk prioritization | Experiment: System 1 improves cost/latency? |
| B1 | Direct Opus 5.5 whole-answer judge, no key | Stronger-model baseline, optional |
| O1 | Answer-key comparison by benchmark script | **Oracle assisted; not a valid reference-free baseline** |

**Success is empirical.** Do not assert P2 is more accurate, faster, or cheaper before measuring it. Laya may improve routing or may add overhead and false negatives. If it does not help, present that finding honestly.

---

## 7. Target runner and experiments

### Target prompt (template)

```text
You are reviewing a SYNTHETIC clinical record for documentation accuracy.
Answer the user's question using the supplied source documents only.
If evidence is absent, contradictory or tentative, say so.
Cite relevant file/line IDs. Provide a concise answer and evidence summary.
Do not invent missing details.

SOURCES:
{all_case_source_files_with_line_ids}

QUESTION:
{case_question}
```

All prompts, responses, settings, and date/time saved. No prompt includes hidden keys. The target is Sonnet 5.5 across the entire task set; not necessarily a model that fails every challenge. If it answers all cases correctly, test the verifier with **independently labeled plausible-wrong answer variants**, identify these as **synthetic wrong-answer bank** rather than actual target mistakes, and preserve the genuine success rate.

### Verifier test bank

For each case, prepare independently annotated:

- 2 correct responses with different prose, ordering, caveats and citations;
- 2 wrong responses, including one *plausibly confident* and one with a subtle omission;
- 1 ambiguous example with the correct expected behavior `INSUFFICIENT_TO_VERIFY` if it is genuinely unclear;
- 1 target model's real first answer.

**Start small:** 5 cases × 4 bank answers + 5 target answers = 25 items. Expand to 6–8 cases and 30–50 total items if time permits. Score cases as clusters; don't pretend 50 variants of six case templates are 50 independent medical scenarios.

### Primary metrics

- **Error recall (catch rate):** wrong answers flagged `ERROR_DETECTED` / all independently labeled wrong answers.
- **False-positive rate:** correct answers incorrectly flagged `ERROR_DETECTED` / all correct answers.
- **Abstention handling:** appropriate `INSUFFICIENT_TO_VERIFY` on genuinely undecidable cases, plus false abstentions.
- **Trap attribution:** proportion of detected errors for which the flagged mechanism matches held-out annotation, judged by teammate review after the fact.
- **Evidence validity:** fraction of source citations that exist and actually support the verifier explanation.
- **Mean and median cost:** from measured token usage and published price rates, per verification.
- **Wall-clock latency:** median and p90 for whole verifier, including networking and Laya if used.
- **Real versus constructed outcomes:** keep tables separate for official target-first-answers and curated correct/incorrect answer bank.

Do **not** measure accuracy solely by matching a hidden string or forcing one exact phrasing. Record `insufficient` separately instead of quietly counting it as correct detection. Small-N results are preliminary and should be described as such.

### Ablations (time permitting)

- P1 without global contradiction sweep.
- P1 without chronology stage.
- P1 versus P2 (Laya enabled).

An ablation matters only if it answers a focused question; don't sacrifice a reliable MVP to run every possible variant.

### Cost methodology

Anthropic lists Haiku 5.5 at **$0.10 input / $0.50 output per million tokens for requests <=100k prompt tokens** (higher rates above 100k); Opus 5.5 at **$4 / $20** per million. Source: https://platform.claude.com/docs/en/models/haiku-5-5/overview and https://platform.claude.com/docs/en/about-claude/pricing .

Compute `cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000`, accounting for caching usage separately when applicable. Thinking tokens are usage, not free. Laya hosted prices are third-party and may change; record the actual charge or measured local compute time. Never substitute per-token pricing ratios for measured **per-verdict** cost ratios.

---

## 8. Data contracts

### `TargetRun`

```json
{
  "run_id": "20261106T180000Z_med_01_sonnet_first",
  "case_id": "med_01",
  "case_version_sha256": "...",
  "model": "claude-sonnet-5-5",
  "timestamp_utc": "2026-11-06T18:00:00Z",
  "exact_prompt": "...",
  "answer_raw": "...",
  "first_attempt": true,
  "stop_reason": "end_turn",
  "input_tokens": 1234,
  "output_tokens": 321,
  "latency_ms": 1480
}
```

### `ClaimVerdict`

```json
{
  "claim_id": "c1",
  "claim_text": "Spironolactone caused the initial potassium rise",
  "status": "CONTRADICTED",
  "severity": "material",
  "citations": [
    {"file": "lab_results.txt", "line_id": "L007", "quote": "..."},
    {"file": "medication_marshal.txt", "line_id": "M004", "quote": "..."}
  ],
  "why": "The initial elevated measurement precedes drug initiation.",
  "uncertainty": "Contribution to later persistent elevation remains unknown."
}
```

### `VerifierRun`

```json
{
  "target_run_id": "20261106T180000Z_med_01_sonnet_first",
  "verifier_variant": "haiku_decomposed",
  "overall": "ERROR_DETECTED",
  "claim_verdicts": [],
  "total_model_calls": 6,
  "input_tokens": 2300,
  "output_tokens": 390,
  "estimated_cost_usd": 0.000425,
  "latency_ms": 2100,
  "raw_model_outputs_path": "runs/verify/..../raw.json"
}
```

The example token counts, costs and latency above are **illustrative placeholders**, not experimental findings. All actual output must derive from logged provider responses.

---

## 9. Stack, repository layout, and CLI

**Recommended:** Python 3.11+, `anthropic`, `pydantic`, `pyyaml`, `httpx`, `rich`, `scikit-learn` or `rank-bm25`, `jinja2`, `pytest`. Use `asyncio` for parallel independent Haiku checks; cap concurrency and retry transient HTTP errors with backoff. `laya` is an **optional** dependency; if local installation and model download are slow, implement the adapter to the explicitly supplied community API, only after verifying a sample call.

```text
medproof/
  README.md
  SPEC.md
  pyproject.toml
  .env.example
  cases/
    med_01/ ... med_06/
  medproof/
    schema.py                 # Pydantic contracts
    load_case.py              # loads chart, restricts key to benchmark-only calls
    source_index.py           # line IDs, chunking, BM25, entity/date extraction
    targets/
      anthropic_runner.py     # one-model, fresh-message, first-answer logging
    verifier/
      claims.py               # Haiku claim extraction
      evidence.py             # retrieval + full-chart contradiction sweep
      check.py                # narrow semantic and timeline checks
      verdict.py              # conservative assembly
      pipeline.py             # parallel execution, fallback, usage metering
    laya_adapter.py           # optional, experimental prioritizer
    baselines.py              # direct Haiku / direct Opus; same evidence, no key
    benchmark.py              # private case keys and human labels
    reporting.py              # aggregate measured metrics
    cli.py
  runs/
    target/                   # exact prompts + first answers
    verifier/                 # raw per-stage outputs + usage
    metrics/                  # no hand-edited result numbers
  report/
    index.html                # generated by reporting.py
  tests/
    test_no_key_leakage.py
    test_source_line_integrity.py
    test_prompt_consistency.py
    test_accept_correct_paraphrases.py
    test_timeline_semantics.py
    test_retrieval_global_conflicts.py
    test_verdict_unresolved.py
    test_cost_meter.py
```

### CLI actions

```bash
python -m medproof.cli validate-cases
python -m medproof.cli run-targets --model claude-sonnet-5-5 --first-only
python -m medproof.cli verify --variant haiku --runs runs/target/
python -m medproof.cli verify --variant haiku-laya --runs runs/target/     # optional
python -m medproof.cli baselines --models haiku,opus --runs runs/target/
python -m medproof.cli evaluate --case-set cases/ --runs runs/
python -m medproof.cli report --output report/index.html
python -m medproof.cli demo --case med_01 --offline
```

`--offline` replays the **real saved API logs**; do not stage a fabricated failure as a live model run. A separate `--live` demo mode may run the verifier on a preserved first target answer, but should have a replay fallback for unreliable internet.

### Environment configuration

```env
ANTHROPIC_API_KEY=
TARGET_MODEL=claude-sonnet-5-5
VERIFY_MODEL=claude-haiku-5-5
HAIKU_EFFORT=low
ENABLE_LAYA=false
LAYA_API_KEY=
LAYA_BASE_URL=https://api.layaaimodel.com
MAX_PARALLEL_CHECKS=5
MAX_CLAIMS_PER_ANSWER=8
```

Do not commit API keys, PHI, or raw environment files.

### Claude SDK implementation notes

Use `output_config={"effort":"low"}` for low-effort Haiku calls as documented by Anthropic; use supported structured output schema methods where compatible with installed `anthropic` SDK. Check current SDK signatures locally rather than hardcoding an unverified method call. The target runner should preserve Sonnet's default configuration and log it. Handle refusal and truncation distinctly. No silent cross-model fallback for official task attempts. Cite docs in README: https://platform.claude.com/docs/en/build-with-claude/effort .

---

## 10. Minimal evaluation page (static HTML)

Build a one-page report; no full frontend framework is required.

**Hero:** MedProof / *A medical AI answer can be plausible and still contradict the record.*

**Left column — Original chart:** file tabs, patient-synthetic banner, line IDs, highlighted contradicting passages; task question in header.

**Center column — Target first answer:** show original unedited text, source citations, first-run timestamp, model ID.

**Right column — Verified findings:** claim cards `SUPPORTED`, `CONTRADICTED`, `NOT_ESTABLISHED`, `CONFLICTED`; source-line explanations, severity, no invented confidence. Clicking a finding highlights matching source lines.

**Below — Experiment:** comparison table for direct Haiku, decomposed Haiku, optionally Haiku+Laya and Opus: **catch rate**, **false alarm rate**, **cost per verdict**, **median latency**. Label whether outcome set is real target first answers or human-labeled answer bank. Include an accessible **counterfactual twin** switch that renders the actual changed chart and *saved verifier result*, not fake results.

Visual design: polished clinical chart aesthetics; restrained whites, charcoal, and muted teal; error red only for actual invalid claims; use plain English throughout. Performance proof matters more than animation.

---

## 11. Build order and work allocation

The event deadline takes priority over the full ambition. Each step below must leave the repository runnable.

### Phase 1 — Reliable minimum viable submission (must ship)

1. **Case team:** Write and peer-validate `med_01` and `med_02`, with source line IDs, ground truths, two correct answer variants and two wrong variants each. Then `med_03`.
2. **Verifier engineer:** Build source index, Haiku claim extraction, retrieval across **all** sources, narrow checks, and conservative verdict for one case.
3. **Experiment/report engineer:** Build one-target first-attempt runner, log every request/response, direct Haiku baseline, and HTML report with one real case result.
4. Verify `NO_ISSUE_FOUND` on at least one correct response and `ERROR_DETECTED` with accurate citation on at least one wrong response. Run unit tests. Package submission.

### Phase 2 — Strong hackathon entry (if time remains)

5. Add `med_03` and 3 twins, keep the target AI identical for every case.
6. Create 20–30 human-labeled correct/incorrect answer-bank items; run pipeline and direct Haiku baseline; calculate measured cost, latency, recall, false alarms.
7. Add **one strong-model judge baseline** only if time and budget permit.
8. Improve visual report and rehearse 3-minute demo.

### Phase 3 — Optional research novelty

9. Integrate Laya, log its classification accuracy on labeled claim/passage pairs, compare P1 vs P2 routing; if it hurts reliability, leave disabled.
10. Add `med_04`–`med_06`, additional fair paraphrases, further blind runs, case-disjoint evaluations, and ablations.

### Team roles (3 people)

- **A — Clinical benchmark owner:** author cases, run blind peer review, validate expected answers, maintain rubric isolation.
- **B — Verification owner:** implement Haiku pipeline and evidence search, optional Laya adapter, line-level explanations.
- **C — Experiment and demo owner:** target runner, baselines, logging, metric computation, report, README, submission zip.

Peer review requires A and at least one of B/C to swap and blindly solve each other's cases. Parallelize only after establishing the shared schema and the first complete case.

### Cut rules

If you have one hour left, stop adding cases and integrations. Ship 3 excellent cases, one real recorded target output per case, accurate expected answers, a verifier with fair correct/wrong checks, a direct cheap baseline, reproducible logs, and a clean three-minute demo. **Drop Laya first**, then Opus baseline, then extra cases. Never drop ground truth review or source-cited explanations.

---

## 12. Three-minute presentation script

**0:00–0:25 — The problem.** “Medical AI often sounds confident but misreads the evidence. A contradiction between two notes or an impossible timeline can reverse what the record justifies.”

**0:25–0:55 — Authentic first answer.** Open an unchanged Sonnet first attempt on `med_01` or `med_02` that actually made a material error. If all targets answered correctly, show a clearly labeled constructed wrong answer and explain that the official first attempts were accurate. Never misrepresent test data.

**0:55–1:55 — What MedProof catches.** Click the erroneous claim, show two source lines (including the line the target omitted), and explain how Haiku decomposed the claim and searched for conflicting evidence. Show not merely the score but the exact reason the claim fails.

**1:55–2:25 — Counterfactual.** Switch one documented fact in the twin. Show the different expected interpretation and the preserved, independently run verifier outcome. Explain why the original objection does or does not apply after the controlled change.

**2:25–2:50 — Results.** Show direct Haiku vs decomposed Haiku (and Laya/Opus only if measured). Read recall **and false-alarm rate**, as well as token-cost and latency, honestly. Name small sample size.

**2:50–3:00 — Close.** “We're not replacing clinicians. We're testing whether evidence-grounded, low-cost checks can catch the errors that confident AI prose hides.”

**Likely judge questions:** How do you know the correct answer? Did your verifier see the key? What happens when the chart disagrees with itself? How often do you falsely reject a correct answer? How many target errors actually occurred? What did Laya improve? Why not just ask Opus? Why does a medication's timing only rule out an *earlier* effect rather than causality as a whole?

---

## 13. Definition of done / acceptance tests

### Hard requirements

- [ ] At least 3 synthetic multi-document medical cases with realistic provenance and stable source line IDs.
- [ ] At least 1 truly subtle reasoning case, not merely unit conversion or number lookup.
- [ ] Every case has independently written, peer-reviewed correct-answer rationale and fair variations.
- [ ] **One target AI model** used on every official task; exact first answers, model IDs, date and prompts logged without modification.
- [ ] Online verifier makes decisions **without answer-key/trap/twin metadata**; automated no-key-leak test passes.
- [ ] It accepts correct variants and flags wrong variants with **valid supporting citation lines**.
- [ ] `CONTRADICTED` and `NOT_ESTABLISHED` have distinct outcomes and explanations.
- [ ] At least 1 actual blind target result and 1 target first-answer result on a twin are preserved.
- [ ] Simple verifier baseline comparison uses same evidence, with measured token cost and false-positive data.
- [ ] Static HTML report and a short reproducible README; project zip passes a clean-install smoke test.
- [ ] Submission complete **before 4:00 p.m.**, leaving time for the demo.

### Stretch requirements

- [ ] Haiku+Laya feature flag, measured against Haiku-only.
- [ ] 6 cases with 3 or more twins.
- [ ] Direct Opus judge baseline.
- [ ] Multi-case per-trap error breakdown, analysis of failed retrieval and Laya routing.

---

## 14. Explicit risks and failure behavior

| Risk | Required response |
|---|---|
| **Target solves all tasks** | Still show actual Sonnet success honestly; verify against independently labeled wrong-answer bank, with synthetic answers clearly marked. |
| **Clinical interpretation disputable** | Revise/cut the case; favor chart-grounded truth over treatment or diagnosis judgments. |
| **Haiku invents a contradiction** | Require both source citations to exist and carry the asserted meaning; otherwise `INSUFFICIENT_TO_VERIFY`. Measure false alarms. |
| **Retrieval misses a hidden contradictory note** | Full-chart sweep for short records; index both supporting and contradictory passages; evaluate source recall. |
| **Laya confidently misroutes important evidence** | Do not let it suppress high-risk checks in default mode. Keep optional and compare false-negative impact. |
| **Extra model calls erase cost savings** | Measure end-to-end per-verdict cost from real tokens; remove stages whose benefit does not justify their cost. |
| **Provider keys fail / rate limits** | Replay authentic logged evaluations offline; no fake responses or fabricated metrics. |
| **Medical privacy / claims** | Synthetic records only; no medical device or clinical-use claims. |
| **Potential scoring leakage** | Separate `cases/.../case.yaml` benchmark secrets from verifier input; add explicit test for forbidden fields and ensure scoring only happens after verdicts are persisted. |
| **Time pressure** | Ship correctness and logs before case count, Laya integration, or UI polish. |

---

## 15. Coding-agent kickoff instruction

> Build MedProof from `SPEC.md`. Start with the exact Phase 1 vertical slice: case loader, one fully reviewed case (`med_01`), one blind Sonnet first-run capture, Haiku claim extraction + evidence retrieval across the full source corpus, narrow grounded verification, conservative verdict, a direct Haiku baseline, and an HTML report. Keep `case.yaml` private answer keys inaccessible to all online verifier components; validate this in tests. Preserve actual source citations, raw prompts and model outputs. Do not invent API responses, outcomes, accuracy numbers, prices, or medical facts. Use only synthetic patient records. Implement one runnable phase at a time with tests and reproducible commands; only add Laya behind a feature flag after the Haiku path runs end to end. Treat the 4:00 p.m. submission deadline as a hard constraint.

---

## References

- Mercor Hack Day — Opening Ceremony, supplied 14-page PDF, particularly pp. 5–10 (challenge, blind protocol, submission, judging).
- Anthropic Haiku 5.5 official overview: https://platform.claude.com/docs/en/models/haiku-5-5/overview
- Anthropic effort docs: https://platform.claude.com/docs/en/build-with-claude/effort
- Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Laya original open-weight model: https://huggingface.co/convaiinnovations/laya
- Laya original GitHub: https://github.com/NandhaKishorM/laya
- **Unofficial**, user-linked Laya API: https://www.layaaimodel.com/docs/api/

**End of spec.**
