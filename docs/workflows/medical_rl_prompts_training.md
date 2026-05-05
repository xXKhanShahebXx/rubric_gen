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

Validation runs use the same preset with `--split val` (no shard flag —
validation is a single contiguous span). The medical pipeline supports
two validation modes; pick whichever matches your goal.

### Mode 1: Pure RRD validation (no retrieval)

The simplest path: each validation example gets its own freshly
discovered rubric bank, scored exactly the same way training was. Use
this when you want a clean apples-to-apples comparison against the
training-set numbers.

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --split val `
  --sample-workers 4 `
  --resume `
  --run-name medical_v47_validation `
  --output-dir artifacts/medical_rl
```

### Mode 2: Retrieval-augmented validation (with relevance filter)

Once all three training shards finish, you can distil the discovered
rubrics into an embedding index, then seed RRD discovery on each
validation example with the top-K nearest training rubrics. A Sonnet
4.5 relevance filter prunes embedding-near but topically off rubrics
(the cancer-X / cancer-Y problem) before they reach the RRD acceptance
filters.

**Step 1 — Build the index** (one-time, ~$0.30 in OpenAI embeddings):

```powershell
python scripts/build_medical_rubric_index.py `
  artifacts/medical_rl/runs/medical_v47_train_shard0 `
  artifacts/medical_rl/runs/medical_v47_train_shard1 `
  artifacts/medical_rl/runs/medical_v47_train_shard2 `
  --out artifacts/medical_rl/rubric_index/medical_v47_index.json
```

The script reads each `examples/<id>.json`, extracts the
`methods.rrd_uniform.rubrics` arrays, dedupes globally on lowercased
text, embeds via `text-embedding-3-small` in batches of 128, and writes
a single JSON index (~37 MB at 6K rubrics). Pass `--dry-run` first if
you want to see the dedupe stats without paying for embeddings.

**Step 2 — Run validation with retrieval + filter:**

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --split val `
  --sample-workers 4 `
  --resume `
  --medical-rubric-index artifacts/medical_rl/rubric_index/medical_v47_index.json `
  --medical-rubric-retrieval-top-k 8 `
  --relevance-filter-strictness conservative `
  --run-name medical_v47_validation_retrieval `
  --output-dir artifacts/medical_rl
```

Knobs you can flip:

- `--medical-rubric-retrieval-top-k N` — number of nearest training
  rubrics to retrieve per validation prompt before filtering. Default 8.
- `--relevance-filter-strictness {conservative,aggressive}` —
  `conservative` (default) drops only `IRRELEVANT` verdicts; UNCERTAIN
  rubrics survive. `aggressive` keeps only `APPLICABLE`.
- `--no-relevance-filter` — keep retrieval, skip the Sonnet filter.
  Useful for measuring the filter's contribution.
- `--relevance-filter-model anthropic:claude-opus-4-7` — swap the
  default Sonnet 4.5 for a stronger (slower, ~5x cost) filter.

**Step 3 — Diagnostics:**

```powershell
# Per-example drop counts and sample reasons
python scripts/inspect_filter_drops.py `
  artifacts/medical_rl/runs/medical_v47_validation_retrieval `
  --samples-per-example 3 `
  --only-with-drops

# Where does the gold response rank, with vs without retrieval?
python scripts/inspect_gold_ranks.py `
  artifacts/medical_rl/runs/medical_v47_validation_retrieval
```

The retrieval+filter debug payload lands at
`methods.rrd_uniform.artifact.retrieval_debug` inside each
`examples/<id>.json`, including:
- the top-K retrieved `rubric_id`s and cosine scores
- the per-criterion filter verdicts (APPLICABLE/IRRELEVANT/UNCERTAIN)
  with their reasons
- the count that actually seeded RRD's initial proposals
- any router or parse errors

The seeded rubrics show up in the per-example RRD bank tagged
`source_stage="initial_seed"` so you can distinguish them from
freshly proposed rubrics. The `rrd_artifact` block on each method
records `seed_rubric_input_count`, `seed_rubric_accepted_count`, and
`seed_rubric_rejected_count` for run-level rollups.

### Mode 3: Pair-preference validation on `medical_gpt41_answers_rl.jsonl`

This is the headline validation path for the current workflow: shard 0
trains the rubric library, and the 4,000-row
`data/medical_gpt41_answers_rl.jsonl` dataset (which carries
`reference_answer_a`, `reference_answer_b`, and a `correct_answer`
label) is used as the ground-truth-comparable benchmark. The metric is
**pair-preference accuracy**: for each row, the pipeline scores both
candidate responses against its rubric bank and the candidate with the
higher whitened-uniform score is the pipeline's "preferred" answer. We
report the percentage of those preferences that match
`correct_answer`. This is the JudgeBench-style metric we couldn't
compute on `medical_rl_prompts.jsonl` because that file has no labels.

**Step 1 — Build the rubric index from your existing shard 0 training
run.** No retraining required; the relevance-filter / retrieval
changes do not touch the training output.

```powershell
python scripts/build_medical_rubric_index.py `
  artifacts/medical_rl/runs/medical_v47_train_shard0 `
  --out artifacts/medical_rl/rubric_index/medical_v47_shard0_index.json
# ~$0.30 in OpenAI embeddings, ~3 min wall clock, ~5,700 unique rubrics.
```

**Step 2 — Pick your parallelism strategy.** Two orthogonal axes,
both already in the codebase:

| Strategy | Concurrent calls / machine | Wall clock | Notes |
|---|---:|---:|---|
| Single machine, `--sample-workers 1` | 4 | ~28-32 hr | Sequential baseline. Don't. |
| Single machine, `--sample-workers 4` | 16 | ~6-8 hr | Training default; tier 2+ comfortable. |
| Single machine, `--sample-workers 8` (recommended single-box) | 32 | **~3-4 hr** | Tier 2+ fine; tier 1 will throttle. |
| 3 machines × `--sample-workers 8` (4-shard split, 3 used) | 32 / machine | **~1.5-2 hr** | Same playbook as the training shards. |
| **4 machines × `--sample-workers 8` (4-shard split, full team)** | **32 / machine** | **~45-60 min** | Cleanest division: 4000 / 4 = 1000 rows / shard. |

`--num-shards N --shard-index I` now works under `--split all`, so the
team-machine sharding pattern from training carries over directly.

**Step 3a — Single-machine command (recommended single-box):**

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_gpt41_answers_rl.jsonl `
  --preset judgebench-v47-medical `
  --split all `
  --num-shards 1 `
  --sample-workers 8 `
  --max-workers 4 `
  --resume `
  --rubrics-only `
  --medical-rubric-index artifacts/medical_rl/rubric_index/medical_v47_shard0_index.json `
  --medical-rubric-retrieval-top-k 8 `
  --relevance-filter-strictness conservative `
  --run-name medical_v47_pair_validation_gpt41 `
  --output-dir artifacts/medical_rl_validation
```

The explicit `--num-shards 1` overrides the preset's training-default
of 3 (which would refuse to load 4,000 rows because 4000 % 3 != 0).

**Step 3b — 4-machine sharded command (one per team member):**

```powershell
# Member A (rows 0..999)
python -m rubric_gen.cli `
  --dataset-path data/medical_gpt41_answers_rl.jsonl `
  --preset judgebench-v47-medical `
  --split all `
  --num-shards 4 --shard-index 0 `
  --sample-workers 8 --max-workers 4 --resume --rubrics-only `
  --medical-rubric-index artifacts/medical_rl/rubric_index/medical_v47_shard0_index.json `
  --medical-rubric-retrieval-top-k 8 --relevance-filter-strictness conservative `
  --run-name medical_v47_pair_validation_gpt41_shard0 `
  --output-dir artifacts/medical_rl_validation

# Member B / C / D: identical, swap `--shard-index 1 / 2 / 3` and
# `--run-name ...shard1 / shard2 / shard3`.
```

**Why these flags:**

- `--split all` (not `--split val`). The new file is its own complete
  4,000-row dataset, not a slice of `medical_rl_prompts`.
- `--rubrics-only` skips the four scoring baselines
  (`compressed_bank`, `production_bank`, `one_shot`,
  `static_healthcare`) and the `direct_judge` baseline. The
  pair-preference metric only consumes `rrd_uniform` /
  `rrd_whitened_uniform`. Without this flag the cost roughly triples.
- `--medical-rubric-index ...` activates retrieval + the relevance
  filter for each pair. Each validation example seeds RRD with the
  top-K nearest (and Sonnet-filter-survived) training rubrics from
  shard 0.
- `--num-shards 4 --shard-index N` partitions the 4,000 rows into 4
  contiguous shards of 1,000 each (4000 % 4 == 0). Add this when
  running across team machines; omit it for single-machine runs.

**Cost estimate (independent of parallelism):** ~$1,750-2,070 total
across all 4,000 rows. Wall clock scales near-linearly with
`sample_workers × num_shards`; the OpenAI router retries any rate-limit
bursts.

**Step 4 — Read the score.** `summary.md` at
`artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt41/summaries/summary.md`
adds a new column when pair examples are present:

```
| Method               | Pair Pref Acc | Pair Pref (correct/n) | Reference Top-1 | Strong>Weak | ... |
|----------------------|--------------:|----------------------:|----------------:|------------:|-----|
| rrd_uniform          | 0.673         | 2691/4000             | 0.000           | 0.987       | ... |
| rrd_whitened_uniform | 0.689         | 2756/4000             | 0.000           | 0.985       | ... |
```

The headline number is **`rrd_whitened_uniform` pair preference
accuracy**. That's the JudgeBench-style ground-truth-comparable metric
for the medical workflow.

**Step 5 — Diagnostics:**

```powershell
# Per-example filter drop counts and reasons
python scripts/inspect_filter_drops.py `
  artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt41 `
  --samples-per-example 3 --only-with-drops

# Per-example pair-preference forensics: which examples did the
# pipeline get right / wrong, with what scores
python -c "import json; from pathlib import Path; r='artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt41'; m=json.loads((Path(r)/'summaries/summary.json').read_text(encoding='utf-8'))['method_metrics']; print(json.dumps(m, indent=2))"
```

After all 4 shards finish, the per-shard `summary.json` files can be
merged with a small script (sum `pair_preference_correct` and
`pair_preference_evaluable` across shards, divide for the unified
accuracy).

### Schema mapping for the pair dataset

Recognised automatically by `dataio.load_examples`:

| JSONL field | Mapped to | Notes |
|---|---|---|
| `id` | `source_id` / `example_id` suffix | Stable identifier. |
| `question` | `task_prompt` | Lowercase variant of `Question`; both work. |
| `reference_answer_a` | `pair_response_a` | Becomes the `pair_a` anchor candidate. |
| `reference_answer_b` | `pair_response_b` | Becomes the `pair_b` anchor candidate. |
| `correct_answer` | `pair_correct_label` | Normalised to `"a"` / `"b"`; unknown values disable pair-preference scoring for that row. |

When pair fields are populated, the loader skips the legacy
`reference_note` / `augmented_note` anchors — the pair anchors take
their place. Backward compat: every other dataset is unaffected
(pair fields default to empty).

---

## v2 runbook — shard 0 retrain + 4k revalidation with the v2 plan changes

This section is the operational follow-up to
[`artifacts/diagnostics/shard0_v2_report.md`](../../artifacts/diagnostics/shard0_v2_report.md)
(Phase 1 forensics on the original shard 0 run) and the code changes
shipped in the same v2 increment (Tier A1–A6 to the rubric proposer / RRD
engine / candidate generation / aggregation, Tier B1–B3 to the validation
retrieval / relevance filter, Tier C1–C2 to the post-hoc cascade).

The headline goals for the 4k pair-preference run are
**strict > 72.17 %**, **raw > 73.67 % (always-B baseline)**, and
**balanced > 65 %** (vs the current 67.23 % / -1.50 pp / 58 %).

### v2 dataset / preset / model assumptions

Same as the original training and §4 validation runs: source datasets are
`data/medical_rl_prompts.jsonl` (training) and
`data/medical_gpt5_b_regen_4k_rl.jsonl` (4k pair validation), preset
`judgebench-v47-medical` (GPT-4o judge + GPT-4o rubric proposer + Sonnet
4.5 / GPT-4o-mini writers), and a single-machine `--sample-workers 8`
layout. Wall-clocks below are at that parallelism on a tier-2 OpenAI
account. Retraining is required because the v2 prompt edits and
multi-sample satisfaction force fresh proposer + judge calls (the
proposer cache key now includes a `prompt_scheme="v2_task_typed"`
sentinel; the satisfaction cache reuses sample-0 calls verbatim).

### Step v2-1 — 100-row smoke retrain (gate before the full retrain)

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --shard-index 0 `
  --num-shards 10 `
  --limit 100 `
  --sample-workers 8 --max-workers 4 --resume `
  --rubric-satisfaction-samples 3 `
  --discrimination-min-pq 0.05 `
  --decomposition-min-recall 0.70 `
  --decomposition-min-discrimination-gain 0.01 `
  --run-name medical_v2_train_shard0_smoke `
  --output-dir artifacts/medical_rl
```

`--shard-index 0 --num-shards 10 --limit 100` evaluates the first 100
rows of shard 0 (rows 0..99 of the source file). Cost ~$150, wall-clock
~1 hr at `sample_workers=8` (the `--rubric-satisfaction-samples 3` flag
roughly triples the per-rubric-call cost vs the v1 run).

After the smoke finishes, gate via the v2 diagnostics. The desired deltas
(vs the v1 numbers in `artifacts/diagnostics/shard0_v2_report.md`) are:

* `overlap` rejection share drops below 60 % (v1: 74.0 %)
* `discrim_useless` bucket drops below 8 % (v1: 11.8 %)
* top-50 4-token prefix concentration drops below 60 % (v1: 72.2 %)
* depth=1 rubric count rises (v1: 18 successful parents on 1k rows ⇒
  proportional smoke target ≥ 4 successful parents on 100 rows)
* per-example anchor-vs-generated diff moves toward 0 (v1: −5.7 pp)

```powershell
python scripts/diagnose_shard0_v2.py `
  --run-dir artifacts/medical_rl/runs/medical_v2_train_shard0_smoke `
  --report-path artifacts/diagnostics/shard0_v2_smoke_report.md
```

### Step v2-2 — full 1000-row shard 0 retrain

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_rl_prompts.jsonl `
  --preset judgebench-v47-medical `
  --shard-index 0 `
  --sample-workers 8 --max-workers 4 --resume `
  --rubric-satisfaction-samples 3 `
  --discrimination-min-pq 0.05 `
  --decomposition-min-recall 0.70 `
  --decomposition-min-discrimination-gain 0.01 `
  --run-name medical_v2_train_shard0 `
  --output-dir artifacts/medical_rl
```

Cost ~$1500, wall-clock ~3-4 hr. The preset still pins `--split train`,
`--train-size 3000`, and `--num-shards 3` so this stays scoped to the
training-dataset's first 1000 rows just like the v1 run.

### Step v2-3 — rebuild the medical rubric index from the v2 shard 0

```powershell
python scripts/build_medical_rubric_index.py `
  artifacts/medical_rl/runs/medical_v2_train_shard0 `
  --out artifacts/medical_rl/rubric_index/medical_v2_shard0_index.json
```

Cost ~$0.30, wall-clock ~3 min. The new index reflects the v2 rubric
library (less templated, more discriminative).

### Step v2-4 — 200-row 4k validation smoke (gate before the full revalidation)

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_gpt5_b_regen_4k_rl.jsonl `
  --preset judgebench-v47-medical `
  --split all --num-shards 1 --limit 200 `
  --sample-workers 8 --max-workers 4 --resume --rubrics-only `
  --rubric-satisfaction-samples 3 `
  --discrimination-min-pq 0.05 `
  --medical-rubric-index artifacts/medical_rl/rubric_index/medical_v2_shard0_index.json `
  --medical-rubric-retrieval-top-k 16 `
  --relevance-filter-strictness conservative `
  --run-name medical_v2_pair_validation_gpt5_b_4k_smoke `
  --output-dir artifacts/medical_rl_validation
```

Cost ~$50, wall-clock ~30 min. Gate: `summary.md` strict pair-preference
accuracy ≥ 70 %. (Strict means score-tie-as-wrong; the cascade hasn't
been applied yet.) If the smoke is below 70 %, stop and inspect before
paying for the full 4k.

The Tier B1 fix (pair-mode filter candidate texts) is already wired into
`rubric_gen/pipeline.py`; no extra flag is required to enable it. The
Tier B2 bump is the explicit `--medical-rubric-retrieval-top-k 16`. To
A/B test Tier B3 (filter strictness), repeat the smoke with one of the
following tail flags:

```powershell
# B3: aggressive strictness (drops UNCERTAIN as well as IRRELEVANT)
... --relevance-filter-strictness aggressive ...
# B3: skip the relevance filter entirely (retrieval-only)
... --no-relevance-filter ...
```

### Step v2-5 — full 4k revalidation

```powershell
python -m rubric_gen.cli `
  --dataset-path data/medical_gpt5_b_regen_4k_rl.jsonl `
  --preset judgebench-v47-medical `
  --split all --num-shards 1 `
  --sample-workers 8 --max-workers 4 --resume --rubrics-only `
  --rubric-satisfaction-samples 3 `
  --discrimination-min-pq 0.05 `
  --medical-rubric-index artifacts/medical_rl/rubric_index/medical_v2_shard0_index.json `
  --medical-rubric-retrieval-top-k 16 `
  --relevance-filter-strictness conservative `
  --run-name medical_v2_pair_validation_gpt5_b_4k `
  --output-dir artifacts/medical_rl_validation
```

Cost ~$700, wall-clock ~8 hr (or distribute across 4 machines using
`--num-shards 4 --shard-index N` per the original §3b layout, dropping
each machine to ~2 hr). The headline number lives in `summary.md` at
`artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k/summaries/summary.md`.

### Step v2-6 — apply the post-hoc cascade (Tier C)

Once the full 4k validation finishes, layer the cheap Tier C fixes via
the existing rescore script. The v2 cascade adds a free format-prior
heuristic before the LLM judge (Tier C1) and supports an anti-tie variant
of the GPT-4o judge (Tier C2):

```powershell
# Strict baseline (no tiebreaker) -- for apples-to-apples with §4.2
python scripts/rescore_with_tiebreaker.py `
  --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k `
  --strategy none --suffix _v2_strict

# Cascade with format prior + uniform + GPT-4o judge (best free-cost option)
python scripts/rescore_with_tiebreaker.py `
  --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k `
  --strategy format_then_uniform_then_judge --workers 8 --suffix _v2_cascade

# Cascade with format prior + uniform + anti-tie GPT-4o judge (Tier C2 best)
python scripts/rescore_with_tiebreaker.py `
  --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k `
  --strategy format_then_uniform_then_anti_tie --workers 8 --suffix _v2_anti_tie
```

Each rescore is ~80 s wall-clock and ~$2.50 in OpenAI judge spend (less
because some rows are now resolved by the format prior without an LLM
call). All three `summary_v2_*.md` outputs land in
`<run-dir>/summaries/` next to the existing `summary.md`.

The headline metric stack to publish:

* **Strict accuracy** (= `summary_v2_strict.md` accuracy column).
  Target ≥ 72.17 % (the v1 cascade-best).
* **Cascade accuracy** (= `summary_v2_anti_tie.md` accuracy column).
  Target ≥ 73.67 % (= always-B baseline).
* **Balanced accuracy** = mean of `recall_a` and `recall_b` from the
  same v2 cascade summary. Target ≥ 65 %.

### Per-family / per-Opus-bucket breakdown

For the same forensics layout as the v1 §4.4 / §4.5 tables, run:

```powershell
python scripts/analyze_pair_validation.py `
  --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k
```

This will emit per-`task_profile_id` accuracy (compare against the v1
`documentation_variants 48.75 %` / `clinical_decision_support 84 %`
spread) and per-Opus-verdict bucket accuracy (compare against v1
`Opus-A 44.67 %` / `Opus-TIE 36.11 %`).

### What the v2 changes actually do (at a glance)

| Tier | File(s) edited | What changed | Why (per shard 0 v2 forensics) |
|---|---|---|---|
| A1 | `rubric_gen/rrd/prompts.py` + `rubric_gen/rrd/engine.py` | Task-type-aware system prompts; ban over-templated openings; force-discrimination rules | Top 4-token prefix `"the note correctly identifies"` covered 9 % of all v1 rubrics; top-50 covered 72.2 % |
| A2 | `rubric_gen/rrd/engine.py` (`run_rrd` decomposition gate) | Adaptive `coverage >= ceil(pool/2)` rule (clamped at the configured threshold) | True decomposition success rate was 1.9 % under the static gate (18 successful parents on 1k rows) |
| A3 | `rubric_gen/candidate_generation.py` | Two boundary candidates per row (terse + padded-uncommitted) | Anchor-vs-generated satisfaction asymmetry was -5.7 pp (rubrics over-rewarded generated) |
| A4 | `rubric_gen/rrd/engine.py` (`evaluate_rubric_on_candidate`) | Multi-sample majority-vote satisfaction; sample 0 reuses legacy cache key | 12.5 % of 4k rows were strict score-ties (worth 0 % accuracy each) |
| A5 | `rubric_gen/rrd/engine.py` (`score_rubric_set`) | Drop rubrics with `min(p,1-p) < 0.05` from weights/rankings | 11.8 % of v1 rubrics had `p(1-p) < 0.05` (effectively zero discrimination) |
| A6 | `rubric_gen/cli.py` + `rubric_gen/config.py` | Surface the new tunables as CLI flags | Lets reproducers / future runs tune without code edits |
| B1 | `rubric_gen/pipeline.py` (`_collect_filter_candidate_texts`) | Pass both pair anchors to the relevance filter on pair-only datasets | 19.8 % of v1 4k rows had zero seed inputs because the filter saw `[""]` |
| B2 | command-line `--medical-rubric-retrieval-top-k 16` | Larger retrieval top-K | More post-filter survivors before RRD acceptance |
| B3 | command-line `--relevance-filter-strictness` / `--no-relevance-filter` | A/B alternative filter policies on the smoke | Quantify the filter's contribution on the new index |
| C1 | `rubric_gen/evaluation/pair_tiebreaker.py` (`format_prior_predict`) | Free heuristic format-prior tiebreaker (list / table / code / terse / explain) | Resolves explicit-format-mismatch ties without an LLM call |
| C2 | `rubric_gen/evaluation/pair_tiebreaker.py` (`direct_judge_pair_anti_tie`) | Anti-tie GPT-4o prompt; separate cache namespace | GPT-4o tied 52.5 % of v1 cascade calls; the anti-tie prompt explicitly tells the judge the rubric stage already tied |
