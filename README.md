# MedProof

**Medical AI can sound right while reasoning incorrectly about a patient's record. MedProof breaks an answer into small evidence checks and uses cheap models to catch contradictions, impossible timelines and unsupported certainty.**

Mercor Hack Day, Georgia Tech, 2026-11-06. Full design: [MedProof_Mercor_Hack_Day_SPEC.md](MedProof_Mercor_Hack_Day_SPEC.md). Team split: [WORKPLAN.md](WORKPLAN.md).

> **Safety.** Every record in this repository is synthetic and labeled `SYNTHETIC CASE — NOT A REAL PATIENT`. MedProof is a research prototype. It is not a medical device and not clinical decision support. It reports "no issue found", never "medically verified".

## Thesis

A single large judge reads a whole answer and returns one opinion. MedProof instead:

1. extracts the answer's check-worthy claims,
2. retrieves supporting **and contradicting** evidence from the whole chart, not only the lines the answer cited,
3. runs narrow, parallel checks with `claude-haiku-5-5` at low effort (support or contradiction, suspected versus confirmed, cause before effect),
4. assembles a conservative verdict: `ERROR_DETECTED`, `NO_ISSUE_FOUND`, `INSUFFICIENT_TO_VERIFY` or `TARGET_REFUSAL`, with exact source-line citations.

The research question is empirical: does decomposition plus evidence search give a better **error recall versus false-alarm versus cost** tradeoff than one holistic judge? We compare against a direct Haiku judge (B0) and optionally a direct Opus judge (B1), both given the same evidence and neither given the key.

## Cases

| Case | Failure mechanism | Question to the target | Twin (one controlled change) |
|---|---|---|---|
| `med_01` | Temporal causality | Can spironolactone explain the initial onset of elevated potassium? | Drug started before the elevation |
| `med_02` | Silent contradiction | Summarize what the chart establishes about drug allergies. | NKDA entry replaced by an allergy acknowledgement |
| `med_03` | Should abstain | Which specific pathogen is confirmed by the record? | Blood culture identifies a pathogen |

Each case has multi-document sources with stable line IDs (`[L006]`), a private key (`case.yaml`), an answer bank of independently labeled correct and wrong answers, and a twin.

**Review status:** the case drafts are marked `needs_second_reviewer`. Under the spec's protocol, one teammate authors, another solves the case blind from the chart alone, and differences are reconciled before freezing. Run `python -m medproof.cli validate-cases` to see the current status.

## How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add ANTHROPIC_API_KEY

python -m medproof.cli validate-cases                 # integrity, review and freeze status
python -m medproof.cli freeze --case med_01           # only after human approval
python -m medproof.cli run-targets --first-only --dry-run   # writes prompts, no API calls
python -m medproof.cli run-targets --model claude-sonnet-5-5 --first-only --yes   # OFFICIAL, once
python -m medproof.cli verify --variant haiku --runs runs/target/
python -m medproof.cli baselines --models haiku,opus --runs runs/target/
python -m medproof.cli evaluate --case-set cases/ --runs runs/
python -m medproof.cli report --output report/index.html
python -m medproof.cli demo --case med_01 --offline
pytest
```

`verify` and `baselines` cover both the real target first answers and the answer bank. `evaluate` writes `runs/metrics/metrics.json` and `metrics.md`. When target answers are not yet labeled, it also writes a label template for humans to fill in. `demo --offline` replays saved logs only. `demo --live` runs the verifier on a preserved answer and falls back to replay if the network fails.

Open `report/index.html` in a browser. Deep links select a case, answer and finding, for example `index.html#case=med_01&answer=<run id>&variant=haiku&finding=1`.

## Blind target protocol

- One target model, `claude-sonnet-5-5`, for every case and twin, at its default configuration (no effort, thinking or output-format overrides, and no cross-model fallback).
- One fresh conversation per case, with the identical neutral prompt from spec section 7. The prompt contains only the chart and the question. `tests/test_prompt_consistency.py` checks that no key text, trap name or answer-bank text appears in it.
- The first response is saved untouched in `runs/target/` with the exact prompt, model, timestamp, request configuration, token usage and the case's SHA-256. There are no retries and no overwrites. Refusals are recorded separately.
- Official runs are refused unless every selected case is frozen, and they require `--yes`.

## Fairness rules

1. The online verifier and the baselines receive only `PublicCase` (question and source text) and the answer text. `tests/test_no_key_leakage.py` enforces this statically and on content.
2. Scoring happens only after verdicts are saved, in `medproof/benchmark.py`, which never calls a model.
3. Correct answers are accepted in any phrasing. Each case's bank includes differently worded correct answers, and the false-alarm rate is a headline metric.
4. Hedges and accurate statements of uncertainty are never errors. `INSUFFICIENT_TO_VERIFY` is reported separately and never counted as a detection.
5. Results from real target first answers and from the constructed answer bank are reported in **separate tables**, and the constructed items are labeled as such everywhere.
6. Cases are frozen with SHA-256 hashes before the official run. Any later edit makes `validate-cases` fail.
7. Cost is computed from logged token usage at published rates, per verdict. Effort settings: https://platform.claude.com/docs/en/build-with-claude/effort. Pricing: https://platform.claude.com/docs/en/about-claude/pricing.

## Results

No official results have been recorded yet. Measured results are generated into [runs/metrics/metrics.md](runs/metrics/metrics.md) and the report page by `evaluate` and `report`. No number in this project is hand-edited. Sample sizes are small and the results are preliminary.

## Repository map

```text
medproof/schema.py              shared contracts (PublicCase, TargetRun, VerifierRun, ...)
medproof/load_case.py           public/private loaders, integrity checks, SHA-256 freeze
medproof/targets/anthropic_runner.py   blind first-attempt runner
medproof/llm.py                 Anthropic wrapper used by every model call (verifier lane)
medproof/verifier/              decomposed Haiku verifier (verifier lane)
medproof/baselines.py           direct-haiku / direct-opus judges
medproof/benchmark.py           post-hoc scoring against keys and human labels
medproof/reporting.py           report/index.html
medproof/cli.py                 command line
cases/                          synthetic charts, private keys, answer banks, twins
runs/                           target first answers, verdicts, labels, metrics
scripts/                        submission packaging and clean-install smoke test
```
