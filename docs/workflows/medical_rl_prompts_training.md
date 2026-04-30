# Medical RL Prompts — 3-shard Training Workflow

## Goal

Run the rubric generation pipeline over `data/medical_rl_prompts.jsonl`
(5,000 medical Q&A rows) and produce per-example rubric banks for the
3,000-row training split, with the work parallelised across three
team members' machines (1,000 rows each). The remaining 2,000 rows are
held out for validation and are *not* touched by this workflow.

## Configuration choice

We use the `judgebench-v47-medical` preset, which is the rubric-judge
core that JudgeBench's v4.7 system (`jb_350_blind_v47_agreement_aware`,
80.57 % single-order on `validation_350`; see
[`judgebench_full_project_report.md`](judgebench_full_project_report.md))
is built on, adapted for single-response medical Q&A:

* `--rubric-model openai:gpt-4o` and `--judge-model openai:gpt-4o`,
  matching v4.7's "GPT-4o is the base judge for every rubric and
  satisfaction call" constraint.
* Multi-model writer pool (`openai:gpt-4o-mini` + Claude Sonnet 4.x
  when keys are present), mirroring v2's multi-model RRD proposer.
* The 3,000 / 2,000 / 3-shard split sized exactly for the team layout.

### What does *not* transfer from v4.7

The bulk of v4.7's lift over v2.1 (69.43 % → 80.57 %) comes from
JudgeBench-domain-specific high-precision verifiers:

* `mmlu_independent_answerer` — fires only on `mmlu-pro` source family.
* `math_independent_solver` (sympy) — fires only on `livebench-math`.
* `reasoning_independent_solver` — fires only on `livebench-reasoning`.
* `code_execution_verifier`, `leetcode_test_runner` — fires only on
  `livecodebench`.
* `pair_verifier` HP override path — requires pair-preference inputs.

None of these have anything to verify on a single-response medical Q&A
row. The preset therefore intentionally leaves them off. The rubric
core that *does* transfer is what scored 69.43 % on JudgeBench (v2.1);
the medical run inherits that, plus whatever lift the medical-domain
candidate diversity and rubric library add later.

## Dataset & split

| Span | Rows | Selector |
|---|---|---|
| Training (sharded) | 3,000 | `--split train --train-size 3000` |
| Validation (held out) | 2,000 | `--split val --train-size 3000 --val-size 2000` |
| Total | 5,000 | row order is preserved from the source JSONL |

The split is **order-preserving** and **deterministic**: each shard
is a contiguous slice of the source file, so all three team members
will produce identical splits without having to share a seed.

For the default `--num-shards 3`:

| Shard index | Train rows (0-indexed) | Owner |
|---:|---|---|
| 0 | `[0,    1000)` | Member A |
| 1 | `[1000, 2000)` | Member B |
| 2 | `[2000, 3000)` | Member C |

`--train-size` must be evenly divisible by `--num-shards`; otherwise
the loader raises rather than silently truncating.

## Per-machine command

Each team member runs the same command, only changing
`--shard-index`. `--run-name` should encode the shard so the artifacts
are easy to merge afterwards.

```powershell
# Member A (shard 0, rows 0..999)
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --shard-index 0 `
  --run-name medical_v47_train_shard0 `
  --output-dir artifacts/medical_rl `
  --max-workers 8

# Member B (shard 1, rows 1000..1999)
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --shard-index 1 `
  --run-name medical_v47_train_shard1 `
  --output-dir artifacts/medical_rl `
  --max-workers 8

# Member C (shard 2, rows 2000..2999)
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --shard-index 2 `
  --run-name medical_v47_train_shard2 `
  --output-dir artifacts/medical_rl `
  --max-workers 8
```

The preset already pins `--split train`, `--train-size 3000`,
`--val-size 2000`, and `--num-shards 3`. Pass any of those flags
explicitly only when you need to override the preset.

## What each run produces

Under `artifacts/medical_rl/runs/medical_v47_train_shard{N}/`:

* `split_manifest.json` — records the dataset path, preset, split,
  shard index, and the first/last `source_id` actually loaded. Use
  this to verify shard assignments before merging.
* `normalized_examples.json` — the exact 1,000 examples this shard
  consumed.
* `examples/<example_id>.json` — per-example rubric bank
  (`rrd_*`, `compressed_bank_*`, `production_bank_*`,
  `one_shot_*`, `static_healthcare_*`, `direct_judge`) and the
  candidate response scores for the existing `response` field.
* `reports/`, `summaries/` — aggregated reporting per run.

The `output-dir` is shared across shards so the per-run cache
(`artifacts/medical_rl/cache/`) is shared by all three members on
their respective machines (i.e. each member has their own copy;
caches are not synced unless you copy them).

## Schema mapping

`load_examples` recognises the medical row schema automatically:

| JSONL field | Mapped to | Notes |
|---|---|---|
| `id` | `source_id` / `example_id` suffix | Stable identifier. |
| `Question` | `task_prompt` | Used as the question text the candidates answer. |
| `response` | `augmented_artifact` (`augmented_note`) | The dataset's existing answer becomes one anchor candidate. |
| `source` | `source` | E.g. `medical_o1_subset_b`. |

`task_profile_id` is left to the existing inference (most rows fall
into `general_instruction_following` or `clinical_decision_support`,
both of which already have RRD discovery dimensions registered in
`compiled/task_profiles.py`).

## Validation

Validation is intentionally out of scope for this workflow. When the
validation pass is needed later, run with the same preset but
`--split val` and a different `--run-name`; no shard flag is needed
because validation is a single contiguous span:

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --split val `
  --run-name medical_v47_validation `
  --output-dir artifacts/medical_rl `
  --max-workers 8
```
