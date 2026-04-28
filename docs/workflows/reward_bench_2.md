# Adapter — `allenai/reward-bench-2`

This document describes how the JudgeBench rubric pipeline is adapted to evaluate
on RewardBench 2 ([dataset card](https://huggingface.co/datasets/allenai/reward-bench-2),
[paper](https://arxiv.org/abs/2506.01937)).

> The full eval result lives in
> [`docs/workflows/reward_bench_2_results.md`](reward_bench_2_results.md) (written
> by the runner once `rb2_full_v47` completes).

---

## What RewardBench 2 measures

| Subset | Items | Task |
|---|---:|---|
| Factuality | 475 | Detect hallucinations in factual user queries |
| Precise IF | 160 | Judge whether output follows precise instructions ("answer without the letter u") |
| Math | 183 | Open-ended math (middle-school physics → college calculus) |
| Safety | 450 | Correctly comply / refuse harm-related prompts |
| Focus | 495 | Detect on-topic answers vs system-prompt-injected drift |
| Ties | 102 | Robustness when multiple correct answers exist (e.g. "name a color of the rainbow") |

For non-Ties subsets every item is **best-of-4** — one `chosen` completion plus
three `rejected` completions. Item is counted correct iff the model rates `chosen`
higher than every rejected. Ties subsets have variable counts per side and use a
weighted accuracy + margin metric.

---

## Why the pipeline doesn't run on RB2 unmodified

The pipeline was built for JudgeBench, which is **pairwise** (A vs B). RewardBench
2 is **best-of-4** and **reward-model-shaped**. The adapter handles three
mismatches:

1. **Best-of-4 → 3 pairwise rows.** Each non-Ties item is expanded into three
   pairs `(chosen, rejected_i)` for `i = 0..2`. Each pair is fed through the
   existing pipeline as if it were JudgeBench, with `chosen` always in
   position A and `label = "A>B"`. An item is correct iff the pipeline
   resolves `A>B` on all three of its pairs.
2. **Subset → family routing.** RB2 has 6 subsets; the pipeline knows 4
   families (`mmlu-pro`, `livebench-reasoning`, `livebench-math`,
   `livecodebench`). The adapter maps subsets to the most semantically
   compatible family so the existing routing / task-profile machinery kicks
   in unchanged:

   | RB2 subset | `source_family` |
   |---|---|
   | Factuality | `mmlu-pro` |
   | Precise IF | `livebench-reasoning` |
   | Math | `livebench-math` |
   | Safety | `mmlu-pro` |
   | Focus | `mmlu-pro` |
   | Ties | `mmlu-pro` |

3. **Ties metric proxy.** The official Ties weighted score uses scalar reward
   model scores. Our pipeline produces pairwise verdicts, so we report a proxy:
   `0.5 × accuracy_term + 0.5 × margin_term` where `margin_term` is the
   fraction of pairs resolved with HIGH-confidence verifier override. This is
   not the leaderboard-quoted Ties number; it's a faithful approximation given
   the pairwise output we have.

---

## Files

| Path | Purpose |
|---|---|
| `rubric_gen/compiled/reward_bench_2_loader.py` | Load HF dataset, expand each item to pairwise rows in `JudgeBenchJoinedExample` shape |
| `rubric_gen/compiled/reward_bench_2_metrics.py` | Aggregate per-pair artifacts into per-subset / leaderboard summaries |
| `rubric_gen/compiled/reward_bench_2_runner.py` | CLI: download → expand → run final-eval → aggregate |
| `tests/test_reward_bench_2_loader.py` | Loader unit tests (10) |
| `tests/test_reward_bench_2_metrics.py` | Metric unit tests (6) |

---

## Reproduction

### Smoke test (~$25, ~10 min, 12 items)

```powershell
python -m rubric_gen.compiled.reward_bench_2_runner `
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 `
  --run-name rb2_smoke_n2 `
  --items-per-subset 2 `
  --max-rejected 3 `
  --max-chosen 1 `
  --max-workers 8
```

### Full eval (~$3K, ~14 hours, all 1865 items)

```powershell
python -m rubric_gen.compiled.reward_bench_2_runner `
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 `
  --run-name rb2_full_v47 `
  --max-rejected 3 `
  --max-chosen 1 `
  --max-workers 16
```

If the run gets interrupted, restart with `--resume` to pick up where it left
off (re-uses already-written per-pair artifacts).

### Re-aggregating without re-running

If you only need to recompute the summary from existing artifacts:

```python
from pathlib import Path
from rubric_gen.compiled.reward_bench_2_metrics import (
    aggregate_pair_artifacts,
    load_artifacts_from_run,
)
run_dir = Path("artifacts/compiled_judgebench_final_eval_runs/rb2_full_v47")
arts = load_artifacts_from_run(run_dir)
summary = aggregate_pair_artifacts(arts)
print(summary.to_dict())
```

---

## Bugs encountered while building the adapter

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Smoke produced only 10 items from 6×2 cap; some subset rows lost | RB2 IDs are unique only **within a subset** ("0" exists in both Factuality and Precise IF). My `pair_id = "rb2-{id}-cN-rM"` collided across subsets and the eval pipeline overwrote artifacts on disk. | Embed the subset slug in `pair_id`: `rb2-{subset_slug}-{id}-cN-rM`. Also group metrics by `(subset, item_id)` not just `item_id`. |
| 2 | All routing fell through to `fallback`; rubric satisfaction returned UNKNOWN; `decision = A=B` everywhere | `judgebench_source_family(source)` re-derives the family from the `source` string and ignores the explicit `source_family` field on the example. My initial `source = "reward_bench_2:Factuality"` had no recognised prefix → fell through. | Prefix the `source` with the family: `source = "{family}:reward_bench_2:{subset}"`. RB2 lineage stays in `metadata.reward_bench_2`. |
| 3 | Candidates arrived in the pipeline with `text_len=0`; discovery skipped the direct pair and only ran on synthetic mutations of empty strings | `join_local_subset_to_official_pairs` overrides `response_A` / `response_B` from the OFFICIAL JSONL, not the local validation file. My `write_official_jsonl` only wrote IDs / labels / subset metadata, not response texts. | Write the FULL pair payload (question, response_model, response_A, response_B, label, original_id, reference_answer) to the official JSONL. |
| 4 | Aggregator found 0 items per subset even though artifacts existed | The pipeline serialises example metadata one level deeper, so my `metadata.reward_bench_2` ended up at `pair.metadata.metadata.reward_bench_2`. | Probe both `pair.metadata.reward_bench_2` and `pair.metadata.metadata.reward_bench_2` in the aggregator. |
| 5 | `load_artifacts_from_run` looked for `validation_350/final/examples/` (hard-coded JudgeBench split name) | The split name in the runner is `--run-name`, not the literal string `validation_350`. | Walk every `examples` directory under the run dir; filter to JSON payloads with a `pair` block. |

All five fixes are in `reward_bench_2_loader.py` and `reward_bench_2_metrics.py`.
The 16 unit tests guard against regressions on each.

---

## Caveats / what's NOT a fair leaderboard number

* **Pipeline runs are not "GPT-4o the reward model".** Just like JudgeBench v4.7,
  the pipeline is a hybrid system (rubric judge + Claude solvers + subprocess
  code execution). The published RewardBench 2 leaderboard is for actual reward
  models, not pipelines that include retrieval / non-OpenAI helpers.
* **Verifier override layer barely fires on RB2.** The MMLU answerer needs
  multiple-choice letter format, which RB2 prompts don't have. The math solver
  fires on ~half of the Math subset. The LeetCode runner essentially never fires
  (RB2 has no LiveCodeBench-style prompts). So the "+11 points from verifier
  overrides" advantage on JudgeBench doesn't transfer here.
* **Best-of-4 cap on Ties.** The leaderboard's Ties metric uses up to ~37
  rejected per item. We cap at 3 rejected per item so the cost stays
  comparable to the other 5 subsets. The Ties number we report is therefore
  not directly comparable to leaderboard Ties.
* **Single-order accuracy** as in JudgeBench v4 — we run each (chosen,
  rejected_i) pair once with `chosen` in position A. We don't currently do an
  order-swapped pass.
