# MedProof: Two-Person Work Plan

The source of truth is `MedProof_Mercor_Hack_Day_SPEC.md` (called "the spec" below). This file splits the work into two lanes that can run in parallel, each driven by its own coding agent.

**Hard deadline: 4:00 p.m. today (2026-11-06).** Aim to submit by 3:45.

| | Person 1: **Verifier** | Person 2: **Benchmark & Experiments** |
|---|---|---|
| Spec role | B (verification owner) | A + C (cases, runner, baselines, report) |
| Owns | `medproof/llm.py`, `medproof/source_index.py`, `medproof/verifier/*`, `medproof/laya_adapter.py` | `cases/*`, `medproof/load_case.py`, `medproof/targets/*`, `medproof/baselines.py`, `medproof/benchmark.py`, `medproof/reporting.py`, `medproof/cli.py`, `report/`, `README.md` |
| Tests | `test_timeline_semantics`, `test_retrieval_global_conflicts`, `test_verdict_unresolved`, `test_cost_meter`, `test_accept_correct_paraphrases` | `test_no_key_leakage`, `test_source_line_integrity`, `test_prompt_consistency` |
| Human-only work | Blind-solve every case Person 2 authors (spec §3, review protocol) | Author the cases and the answer bank, run the official target pass **once**, and rehearse the demo |

Rule: **only edit files in your own lane.** Changes to shared files (Step 0) need a message to the other person first.

---

## Step 0: Shared contracts (both people together, 20 min, before splitting)

Commit these to `main` before either agent starts. They are the only interface between the two lanes.

1. **`pyproject.toml`**: Python ≥3.11 with `anthropic`, `pydantic`, `pyyaml`, `httpx`, `rich`, `rank-bm25`, `jinja2`, `pytest`. Also add `.env.example` (spec §9) and a `.gitignore` covering `.env` and `runs/` scratch.
2. **`medproof/schema.py`**: Pydantic models copied from spec §8, plus these:
   - `SourceLine {file, line_id, text}`
   - `PublicCase {case_id, question, sources: dict[str, str], source_sha256}`. This is the **only** case object the verifier and the target runner ever see.
   - `Citation {file, line_id, quote}`
   - `ClaimVerdict`, `VerifierRun`, `TargetRun` (exactly as in spec §8)
   - Enums: `Overall`, `ClaimStatus`, `Trap`
3. **Verifier entry point** (Person 1 implements it; Person 2 only calls it):
   ```python
   # medproof/verifier/pipeline.py
   def verify(case: PublicCase, answer: str, variant: str = "haiku",
              target_run_id: str | None = None) -> VerifierRun: ...
   ```
   Commit a **stub** that returns `overall="INSUFFICIENT_TO_VERIFY"`, so Person 2's end-to-end flow runs from minute one.
4. **LLM wrapper**: commit a stub signature, which Person 1 implements:
   ```python
   # medproof/llm.py
   async def call(model, system, user, *, effort=None, schema=None, max_tokens=1024) -> LLMResult
   # LLMResult: text | parsed, input_tokens, output_tokens, cache tokens, stop_reason, latency_ms, cost_usd, raw
   ```
   Everyone uses this wrapper for every Anthropic call (the target runner, the verifier and the baselines), so cost metering is consistent.
5. **Case format**:
   - Each source line starts with `[P001]`, `[M001]`, `[L001]`, etc.
   - `cases/<id>/case.yaml` is **private**.
   - `load_case.load_public(case_dir) -> PublicCase` and `load_case.load_private(case_dir)` are the only loaders.
6. **Fixture**: `cases/_fixture/` is a tiny fake chart (3 files, about 10 lines each) with one obvious contradiction. Person 1 develops against it until `med_01` lands.

---

## Person 1: Verifier lane

**Goal:** `verify()` returns honest, cited verdicts without ever seeing the key.

| # | Deliverable | Spec | Done when |
|---|---|---|---|
| 1.1 | `llm.py`: async Anthropic wrapper, structured output, `effort=low` on Haiku, retry with backoff, concurrency cap (`MAX_PARALLEL_CHECKS`), and cost from the pricing table (Haiku $0.10/$0.50, Opus $4/$20, Sonnet from the pricing page). Refusals and truncation are surfaced and never hidden. | §5, §7, §9 | `test_cost_meter` passes |
| 1.2 | `source_index.py`: parse line IDs into `SourceLine`, extract dates/timestamps, BM25 retrieval with top-k 4–8, and a full-chart dump for short records | Stage 0 | Works on the fixture |
| 1.3 | `verifier/claims.py`: Haiku extracts 3–8 atomic claims with `kind`, `cited_lines` and `importance`. Few-shot examples come from a **non-medical** domain. | Stage 1 | — |
| 1.4 | `verifier/evidence.py`: for each claim, gather cited lines ±2, top supporting and contradicting passages, and temporal events. Also runs the **global sweep** (one call over the whole chart) to find omitted contradictions. | Stages 2 & 4 | `test_retrieval_global_conflicts` passes: it finds an NKDA-vs-allergy conflict even when the answer cites only the NKDA line |
| 1.5 | `verifier/check.py`: narrow parallel Haiku checks (support/contradict/not established, suspected-vs-confirmed, same patient, cause-before-effect), plus Python date comparison | Stages 3 & 5 | `test_timeline_semantics` passes: a drug started on Day 3 cannot explain an onset on Day 2, but the check must **not** conclude "not the cause of persistence" |
| 1.6 | Citation validation: every line ID exists and every quote matches the source exactly or after normalization; otherwise the claim is downgraded | Stage 3 | — |
| 1.7 | `verifier/verdict.py`: conservative assembly. If checks disagree or a citation is invalid, return `INSUFFICIENT_TO_VERIFY`. Hedges never cause a failure. `CONTRADICTED` and `NOT_ESTABLISHED` stay distinct. | Stage 6 | `test_verdict_unresolved` and `test_accept_correct_paraphrases` pass |
| 1.8 | `verifier/pipeline.py`: orchestration, raw per-stage outputs saved to `runs/verifier/<target_run_id>/<variant>/raw.json`, and totals for usage, cost and latency | §8 | Real `verify()` replaces the stub |
| 1.9 *(Phase 3, optional)* | `laya_adapter.py` behind `ENABLE_LAYA` as variant `haiku-laya`; it only **prioritizes** claims and never skips one | §6 | Drop this first if short on time |

**Person 1 agent kickoff prompt:**
> Read `MedProof_Mercor_Hack_Day_SPEC.md` §5, §8, §9 and `WORKPLAN.md` (Person 1 lane). Implement items 1.1–1.8 in order, in only the files your lane owns. Do not modify `schema.py` or `load_case.py`. The verifier receives only `PublicCase` and the answer text; never import or read `case.yaml`, `answer_bank.yaml` or anything under `twin/`. Develop against `cases/_fixture/` until `cases/med_01/` exists. All Haiku calls use `claude-haiku-5-5` with `effort=low` via `medproof/llm.py`. Check the installed `anthropic` SDK's structured-output signature locally before using it. Never fabricate model outputs, quotes or numbers. Write the listed tests and keep `pytest` green after each item. Commit after each item.

---

## Person 2: Benchmark & Experiments lane

**Goal:** frozen, peer-reviewed cases; one official blind Sonnet pass; baseline and metrics; report; submission zip.

| # | Deliverable | Spec | Done when |
|---|---|---|---|
| 2.1 | `load_case.py`: `load_public` / `load_private`, plus a SHA-256 freeze of sources and key | §3 | `test_no_key_leakage` passes (the verifier package never imports `load_private` or reads `case.yaml`) and `test_source_line_integrity` passes |
| 2.2 | **Cases `med_01`, `med_02`, then `med_03`**, each with sources, `case.yaml` and an `answer_bank.yaml` (2 correct, 2 wrong, plus 1 ambiguous if fair). The agent drafts; **the human authors and approves**; Person 1 blind-solves before freezing. Push `med_01` sources **early** (target 14:15) so Person 1 can test on real data. | §2, §3, §7 | `validate-cases` passes and review is logged in `case.yaml` |
| 2.3 | Twins `med_01_twin`, `med_02_twin`, `med_03_twin` (one controlled fact changed) | §2 | Twin keys reviewed |
| 2.4 | `targets/anthropic_runner.py`: neutral prompt template (§7), `claude-sonnet-5-5` with default config, fresh call per case and twin, **first answer saved untouched** as a `TargetRun` in `runs/target/`, refusals recorded separately, no retries | §4 Path A | `test_prompt_consistency` passes. **Run officially only after the cases are frozen. Running it once is final.** |
| 2.5 | `baselines.py`: B0 is a direct whole-answer Haiku judge using the same `PublicCase` and no key. B1 (Opus) is optional. | §6 | Each produces a `VerifierRun` with variant `direct-haiku` or `direct-opus` |
| 2.6 | `benchmark.py`: **after** verdicts are saved, score them against the private keys and human labels. Computes recall, false-alarm rate, abstention handling, evidence validity, cost (mean/median) and latency (median/p90). Real target answers and bank answers are reported in **separate tables**. | §4 Path C, §7 | Numbers come only from `runs/` |
| 2.7 | `reporting.py` builds `report/index.html` with three columns (chart / first answer / findings), the experiment table and a twin toggle, all from saved runs | §10 | — |
| 2.8 | `cli.py`: `validate-cases`, `run-targets`, `verify`, `baselines`, `evaluate`, `report`, `demo --offline` (replays real logs) | §9 | — |
| 2.9 | README (thesis, how to reproduce, fairness rules, results table, safety disclaimer) and the submission zip with a clean-install smoke test | §0, §13 | Submitted |

**Person 2 agent kickoff prompt:**
> Read `MedProof_Mercor_Hack_Day_SPEC.md` §2–4, §7–10, §13 and `WORKPLAN.md` (Person 2 lane). Implement items 2.1, 2.4–2.8 in only the files your lane owns, calling `medproof.verifier.pipeline.verify()` as a black box (it may be a stub). For 2.2–2.3, draft synthetic charts with `SYNTHETIC CASE — NOT A REAL PATIENT` banners and stable line IDs, but mark every `case.yaml` as `review_status: needs_second_reviewer`; a human finalizes them. Never call the target model for the official run without explicit instruction; build and test the runner against `cases/_fixture/` with a `--dry-run` flag. Never fabricate model outputs, metrics or prices; every number in the report must come from `runs/`. Write the listed tests and keep `pytest` green. Commit after each item.

---

## Timeline and sync points (from 1:30 p.m.)

| Time | Person 1 | Person 2 | Sync |
|---|---|---|---|
| 13:30–13:50 | Step 0 together | Step 0 together | **Contracts merged to `main`** |
| 13:50–14:30 | 1.1, 1.2, 1.3 | 2.1, draft `med_01` sources, 2.4 (dry run) | 14:15: `med_01` sources pushed |
| 14:30 | Blind-solve `med_01` (10 min, human) | Draft `med_02` | **Sync 1:** freeze `med_01` |
| 14:30–15:10 | 1.4–1.8; real `verify()` merged by 15:00 | `med_02`, `med_03`, answer banks, 2.5, 2.6 | Person 1 blind-solves `med_02`/`med_03` as they land |
| 15:10 | — | **Official Sonnet run** on frozen cases + twins | **Sync 2:** run pipeline + baseline on everything |
| 15:10–15:35 | Fix false alarms; optionally Laya | 2.7 report, 2.8 demo, README | No case/key edits after the official run |
| 15:35–15:45 | Smoke-test the zip | Package and submit | **Submitted** |
| 15:45–16:00 | Rehearse the 3-min demo together | | |

**Git:** each person works in their own branch or worktree (`p1-verifier`, `p2-bench`) and merges to `main` at each sync. File ownership above means merges should not conflict.

## Cut order (spec §11)
First Laya, then the Opus baseline, then `med_04`–`med_06`, then the third twin. **Never cut** blind review, source-cited explanations, the no-key-leak test or honest labeling of real versus constructed results.
