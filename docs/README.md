# Rubric System Docs

## Purpose
This directory contains the stable design documents for the compiled rubric system that complements the
existing RRD-style pipeline in this repository.

## Documents
- `spec/compiled_rubric_system.md`
  - Canonical architecture for the compiled rubric stack, artifact boundaries, scoring model, and promotion gates.
- `spec/schema_contracts.md`
  - Concrete payload contracts for ontology, note-family specs, compiled case rubrics, evaluation records, and adjudication records.
- `spec/examples/` (generated)
  - Example JSON artifacts produced by the starter scaffold (`python -m rubric_gen.compiled.demo` from the repo root).
- `workflows/discovery_workflow_100_500.md`
  - Runbook for the `100`-example design phase and `500`-example pilot rollout.
- `workflows/judgebench_train_only_protocol.md`
  - Leak-safe JudgeBench workflow: train-only development on the `80`, frozen-policy locking, and one-shot final evaluation on the `350`.
- `adjudication/expert_adjudication_packet.md`
  - Template and operating guidance for clinician-facing ambiguity review.

## How To Read These Docs
Recommended order:

1. `spec/compiled_rubric_system.md`
2. `spec/schema_contracts.md`
3. `workflows/discovery_workflow_100_500.md`
4. `workflows/judgebench_train_only_protocol.md`
5. `adjudication/expert_adjudication_packet.md`

## Starter implementation

The Python package `rubric_gen/compiled/` implements dataclasses matching `schema_contracts.md`, JSON
serialization helpers, a labeled starter compiler (`compiler.py`), and a non-LLM heuristic judge for
demos. Run `python -m rubric_gen.compiled.demo` to emit sample artifacts into `spec/examples/`.

For an end-to-end **starter** pilot scaffold (slice split, contrast candidates, heuristic and/or LLM
evaluation, artifacts under `artifacts/compiled_runs/`), see the root `README.md` section *Starter compiled pilot runner*
or run `python -m rubric_gen.compiled.pilot_runner --smoke`. Use `--judge-mode llm` or `both` for the prototype
LLM analytic judge (`rubric_gen.compiled.llm_judge`); optional `--judge-model provider:model` overrides env defaults.

For **starter local rubric discovery** (strong/weak pairs → proposed atomic criteria, one LLM call per pair),
run `python -m rubric_gen.compiled.discovery_runner --smoke` (see root `README.md`). Artifacts live under
`artifacts/compiled_discovery_runs/`; each pair now records `raw_proposals`, a deterministic pair-grounding
filter result, surviving `proposals`, and `rejected_proposals`. Standalone discovery still does not update the
ontology automatically.

For the **provisional iterative runner** (`python -m rubric_gen.compiled.iterative_runner --smoke --judge-mode llm`),
the design-slice discovery output is passed through that pair-grounding filter before merged proposals are converted
into provisional ontology templates. During promotion, overlapping structure/section discoveries are collapsed into
one broader note-family scaffold template, and overlapping symptom-detail discoveries from the symptom-strip
mutation are collapsed into one broader completeness template. Provisional discovered severities are also capped so
they cannot promote themselves into catastrophic strength before human review. Repeated inflate-certainty
discoveries are likewise collapsed into broader note-family certainty templates before the compiled pilot is
replayed on the full split. The current mutation set also targets follow-up timing, return precautions,
medication plans, intervention/procedure plans, planned diagnostic testing, and assessment-plan reasoning
gaps, and repeated variants from those families are compressed into narrower management-plan or
diagnostic-reasoning templates before promotion.
