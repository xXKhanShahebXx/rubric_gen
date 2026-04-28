# RRD Rubric Pipeline

This repository currently contains two closely related layers:

- a runnable `RRD`-style pipeline for rubric generation and note evaluation
- a new `compiled rubric system` design documented under `docs/` for a harder-gated, note-family-aware
  medical dialogue-to-note evaluator

The existing runnable pipeline includes:

- dataset normalization
- hybrid candidate-note generation
- recursive rubric decomposition and filtering
- uniform and whitened-uniform weighting
- baseline comparisons
- per-example artifacts and summary reports

## Usage

```bash
python run_pipeline.py --limit 5 --run-name smoke
```

You can also override model routing explicitly:

```bash
python run_pipeline.py \
  --writer-model openai:gpt-4.1-mini \
  --writer-model anthropic:claude-sonnet-4-20250514 \
  --writer-model together:meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --rubric-model openai:gpt-4.1 \
  --judge-model openai:gpt-4.1-mini
```

Artifacts are written under `artifacts/runs/<run-name>/`.

## Compiled Rubric Docs

The compiled rubric system docs live under `docs/`.

- `docs/README.md`
- `docs/spec/compiled_rubric_system.md`
- `docs/spec/schema_contracts.md`
- `docs/workflows/discovery_workflow_100_500.md`
- `docs/workflows/judgebench_train_only_protocol.md`
- `docs/adjudication/expert_adjudication_packet.md`

### JudgeBench train-only protocol

For the leak-safe JudgeBench workflow, use the `train-only` and `final-eval` commands from
`rubric_gen.compiled.judgebench_eval_runner`:

```bash
# Mechanism development on the 80 only
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset data/judgebench_80_human.json \
  --protocol-mode generic_baseline \
  --train-split-name train_80 \
  --max-workers 8 \
  --run-name jb_train_only_v0

# One-shot held-out evaluation on the 270 using the locked policy
python -m rubric_gen.compiled.judgebench_eval_runner final-eval \
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_train_only_v0 \
  --validation-dataset data/judgebench_270_generated.json \
  --validation-split-name validation_270 \
  --max-workers 8 \
  --run-name jb_final_eval_v0
```

`final-eval` now hides `reference_answer` by default on the held-out split, so the reported `validation_270`
number is a blind benchmark score unless you explicitly opt back into `--allow-reference-answer`.

`train-only` also keeps fold-dev / OOF scoring blind by default now. The train side may still use the `80`
references unless you pass `--no-train-reference-answer`.

See `docs/workflows/judgebench_train_only_protocol.md` for the full workflow and leakage guardrails.

### Compiled rubric scaffold (Python)

Starter schemas, a minimal compiler, and a heuristic demo live under `rubric_gen/compiled/`. Generate
example JSON artifacts from the bundled sample dataset:

```bash
python -m rubric_gen.compiled.demo
# or, if the package is installed:
rubric-compiled-demo
```

Outputs are written to `docs/spec/examples/` (ontology, note-family spec, one case rubric, and sample evaluations).

### Starter compiled pilot runner (scaffold)

A lightweight end-to-end **starter** runner (not full discovery/adjudication) loads a dataset, splits rows into
design / validation / pilot slices, compiles case rubrics, builds original + synthetic contrast candidates,
and runs the heuristic judge. Training-style subsets (`gold_sft`, `repair`, `do_not_train`) include **original
notes only**; synthetic mutations are evaluated for contrast metrics but excluded from those subsets.

```bash
# Small smoke run (8 examples, 1+1+6 slice split)
python -m rubric_gen.compiled.pilot_runner --smoke --csv

# Starter 100/500-style counts on the bundled 500-row sample (truncates to first 500 rows)
python -m rubric_gen.compiled.pilot_runner --dataset data/sample_100_aci_400_agbonnet.json --design 100 --validation 100 --pilot 300 --csv

# LLM analytic judge (single call per candidate; requires API keys in env — see rubric_gen.config)
# Optional: RUBRIC_GEN_COMPILED_JUDGE_MODEL=openai:gpt-4.1-mini  (or use --judge-model provider:model)
python -m rubric_gen.compiled.pilot_runner --limit 1 --design 1 --validation 0 --pilot 0 --judge-mode llm

# Heuristic + LLM side-by-side; training-style subsets stay heuristic-driven; see summaries/review_queue.json
python -m rubric_gen.compiled.pilot_runner --limit 2 --design 1 --validation 0 --pilot 1 --judge-mode both

# After install:
rubric-compiled-pilot --smoke
```

Artifacts: `artifacts/compiled_runs/<run_name>/` with per-example JSON under `examples/`, subset lists,
`run_summary.json`, and `summaries/review_queue.json` (follow-up when the LLM is uncertain, confidence is low,
or judges disagree). LLM responses are cached under `cache/compiled_llm_judge.jsonl` unless `--no-llm-cache`.
Optional `candidate_evaluations.csv` with `--csv`.

### Starter local rubric discovery (scaffold)

A **starter** discovery pass proposes **local atomic criteria** from **strong vs weak** note pairs (reference
→ augmented → truncated as the strong anchor; synthetic mutations from `mutations.py` as weak contrasts when
available). One LLM call per pair; responses are cached like the compiled LLM judge (`RUBRIC_GEN_COMPILED_JUDGE_MODEL`
or default judge discovery). Pair artifacts include the raw LLM proposals plus a deterministic pair-grounding filter
that keeps only proposals aligned with the observed strong/weak delta; filtered-out rows are retained as
`rejected_proposals` for auditability. Standalone discovery still does **not** update the compiled ontology
automatically — it writes artifacts for manual or later automated incorporation.

```bash
# Tiny smoke (1 example, 1 pair, ≤4 criteria; requires API keys in env)
python -m rubric_gen.compiled.discovery_runner --smoke

# After install:
rubric-compiled-discovery --smoke
```

Artifacts: `artifacts/compiled_discovery_runs/<run_name>/` with per-example JSON under `examples/`,
`summaries/merged_proposals.json`, `summaries/run_summary.json`, and `cache/compiled_discovery.jsonl` (unless `--no-cache`).

### Provisional iterative runner (closed-loop scaffold)

A **starter** harness runs **local discovery on the design slice**, converts merged proposals into **additive
provisional criterion templates** (stable `discovered__…` ids), writes **base vs augmented** ontologies to disk,
and **reruns the compiled pilot** on the full split using the augmented ontology. This is explicitly a
**provisional closed-loop scaffold** — not a validated recursive system, not clinical sign-off, and not
automatic semantic vetting of discovered criteria. Only mutation-grounded discovery proposals survive into the
merged proposal pool that feeds the augmented ontology, and overlapping structure/section discoveries are
collapsed into one broader note-family scaffold template at promotion time. Likewise, overlapping symptom-detail
discoveries from the symptom-strip mutation are collapsed into one broader completeness template instead of
proliferating as several near-duplicate promoted checks. Provisional discovered severities are capped below
catastrophic until human review, even if a single local discovery labels itself as `hard_gate`. Repeated
inflate-certainty discoveries are also collapsed into broader note-family certainty templates rather than
accumulating multiple near-duplicate certainty rules. The contrast set now also includes targeted synthetic
mutations for follow-up timing, return precautions, medication plans, intervention/procedure plans, planned
diagnostic testing, and assessment-plan reasoning gaps, with promotion-time consolidation into narrower
management-plan and diagnostic-reasoning template families when repeated variants appear across cases.

```bash
# Tiny smoke (1 design example for discovery + 1 pilot example; requires API keys for LLM paths)
python -m rubric_gen.compiled.iterative_runner --smoke --judge-mode llm

# After install:
rubric-compiled-iterative --smoke --judge-mode llm
```

Artifacts: `artifacts/compiled_iterative_runs/<run_name>/` with `run_summary.json` at the top level,
`discovery/` (same layout as standalone discovery), `ontology/` (`base_ontology.json`, `augmented_ontology.json`,
`selected_discovered_templates.json`), and `pilot/` (nested compiled pilot run with `examples/`, `summaries/`, `cache/`).
