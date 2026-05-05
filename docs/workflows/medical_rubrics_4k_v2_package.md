# Medical 4k Rubric Package v2 — Schema and RL Training Guide

This doc describes the **v2 rubric bundle** distilled from
`artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k`
(the full 4k revalidation run from the v2 plan — see
[`medical_pair_judge_v2_results.md`](medical_pair_judge_v2_results.md)
for the headline numbers).  It is the v2 successor to
[`medical_rubrics_4k_package.md`](medical_rubrics_4k_package.md) and is
drop-in compatible with the same `id`-based join: every reader / loader
written against the v1 package works on v2 by changing the file stem
from `medical_gpt5_b_regen_4k` to `medical_gpt5_b_regen_4k_v2`.

The v2 cascade hit **76.55 % accuracy / 69.93 % balanced** on the same
4k pairs that v1 hit 72.17 % / 63.5 % on, so any reward model trained
on the v2 bundle inherits a stronger upstream label distribution.

---

## 1. Files in the v2 package

| File | Rows | Bytes | Granularity |
|---|---:|---:|---|
| `data/medical_gpt5_b_regen_4k_rl.jsonl` | 4,000 | ~5 MB | one row per pair sample (slim file, **shared with v1**) |
| `data/medical_gpt5_b_regen_4k_v2_rubrics.jsonl` | **4,000** | ~22 MB | one row per pair sample with v2 rubrics + production-bank summary |
| `data/medical_gpt5_b_regen_4k_v2_rubric_evaluations.jsonl` | **234,659** | ~402 MB | one row per (sample, rubric, candidate) GPT-4o YES/NO judgment |

Compared to the v1 package (`*_rubrics.jsonl` 21 MB / 4000 rows, evals
171,089 rows / 273 MB), v2 ships **37 % more evaluation rows** -- the
extra rows come from (a) the new `synthetic` candidate role (boundary
terse + padded-uncommitted candidates from Tier A3, ~46 k extra
evaluations) and (b) slightly larger banks per row.  The slim file
(`medical_gpt5_b_regen_4k_rl.jsonl`) is **the same file** as v1 -- the
gold labels did not change, only the rubric library and per-rubric
evaluations did.

All three files share the same `id` field (e.g. `0006012-c2220aa60dd9`)
and the rubrics file is written in the **same row order** as the slim
file, so they can be `zip()`'d row-for-row at training time without an
explicit join.

The original artefacts the package was distilled from are still on disk:

```
artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k/
├── examples/<id>.json             (4000 per-row pipeline dumps)
├── reports/production_banks.csv   (cross-corpus canonical bank)
└── summaries/
    ├── summary.md                 (strict baseline, 65.88 %)
    ├── summary_v2_strict.md       (= summary, written by rescore)
    ├── summary_v2_v1_cascade.md   (apples-to-apples vs v1 cascade, 71.35 %)
    └── summary_v2_anti_tie.md     (the headline -- 76.55 % cascade, 69.93 % balanced)
```

---

## 2. Schema — `medical_gpt5_b_regen_4k_v2_rubrics.jsonl`

One JSON object per line.  Schema is identical to v1's
`medical_gpt5_b_regen_4k_rubrics.jsonl`; only the contents (rubric texts
and bookkeeping fields) differ.

```jsonc
{
  "id": "0006000-ac3b664c73bd",
  "example_id": "general_instruction_following__0006000-ac3b664c73bd",
  "source": "general_instruction_following",
  "question": "...",
  "reference_answer_a": "...",   // gpt-4o original
  "reference_answer_b": "...",   // gpt-5 regen
  "gold_label": "b",             // 'a' or 'b' (Opus 4.7 with tie-policy=a)
  "correct_answer": "reference_answer_b",

  "rubrics": [
    {
      "rubric_id": "general_instruction_following__0006000-ac3b664c73bd__rubric_0_0_0",
      "text": "The response includes potential treatment options, such as voice therapy or further medical intervention, if nerve damage is confirmed.",
      "source_stage": "initial_seed",       // initial | initial_seed | decomposition
      "depth": 0,
      "parent_id": null
    },
    ...
  ],
  "rubric_count": 9,                        // mean 7.53 in v2 (vs 7.13 in v1)

  "production_bank": [   // cross-corpus canonicalised items this row contributed to
    {
      "production_rubric_id": "production_0",
      "group_id": "monitoring_followup",
      "label": "Monitoring and Follow-Up",
      "family": "assessment_and_plan",
      "canonical_text": "If monitoring instructions or follow-up plans were discussed, ...",
      "conditionality": "if_discussed",
      "importance_tier": "major",
      "action_taken": "kept",
      "source_member_count": 1,
      "coverage_count": 0.0,
      "discrimination_score": 0.0
    }
  ],

  "rrd_artifact": {
    "initial_rubric_count": 9,
    "initial_seed_rubric_count": 7,
    "seed_rubric_input_count": 16,         // v2 retrieval top_k=16 (vs v1's 8)
    "seed_rubric_accepted_count": 7,
    "seed_rubric_rejected_count": 9,
    "final_rubric_count": 9
  },

  "ranking": [
    {"candidate_id": "...__pair_b",      "rank": 1, "score": 0.39021},
    ...
  ]
}
```

Field notes (v2-specific):

* **`source_stage`** -- v2 introduces a noticeably higher proportion of
  `initial_seed` rubrics because Tier B1 (pair-mode filter fix) and
  Tier B2 (top-k=16) recover ~196 of v1's 793 zero-seed rows.  Mean
  seed-input rose 28 % (2.44 → 3.13).
* **`depth`** -- v2 has 94 depth=1 rubrics in the bank (vs v1's 36),
  driven by the Tier A2 adaptive coverage gate.  Still a small share of
  the bank (~2.3 %), but easier to spot in production-bank mapping.
* **`rrd_artifact.seed_rubric_*_count`** -- diagnostic for the
  v2 retrieval health.  `seed_rubric_input_count` will be up to 16
  (vs v1's max 8) thanks to the larger retrieval window.

## 3. Schema — `medical_gpt5_b_regen_4k_v2_rubric_evaluations.jsonl`

One JSON object per (sample, rubric, candidate) triple.  Same shape as
v1; the new field versus v1 is the **`synthetic` candidate role**.

```jsonc
{
  "id": "0006000-ac3b664c73bd",
  "example_id": "general_instruction_following__0006000-ac3b664c73bd",
  "rubric_id": "general_instruction_following__0006000-ac3b664c73bd__rubric_0_0_0",
  "rubric_text": "The response includes potential treatment options, ...",
  "candidate_id": "general_instruction_following__0006000-ac3b664c73bd__boundary_terse",
  "candidate_role": "synthetic",                // NEW v2 role -- boundary candidates
  "candidate_origin": "synthetic",
  "candidate_text": "Voice therapy.",
  "satisfied": false,                            // GPT-4o YES/NO verdict
  "reasoning": "The response is too brief to mention treatment options ..."
}
```

`reasoning` is truncated to `--reasoning-char-limit` characters (default
600) when packaging.

### `candidate_role` values (v2)

| `candidate_role` | What it is | YES rate (first 50k rows) | v1 reference YES rate |
|---|---|---:|---:|
| `pair_anchor_a` | Original A response (GPT-4o) — `reference_answer_a` | 32.8 % | 33.3 % |
| `pair_anchor_b` | Regenerated B response (GPT-5) — `reference_answer_b` | **47.0 %** | 51.7 % |
| `synthetic` (NEW) | Boundary candidates -- `boundary_terse` + `boundary_padded_uncommitted` | 29.2 % | n/a (new in v2) |
| `generated_direct` | GPT-4o-mini, direct system prompt | 17.8 % | 16.0 % |
| `generated_explained` | Claude Sonnet 4.5, "explain your reasoning" | 44.6 % | 47.6 % |
| `generated_comprehensive` | longer-form variant | 45.4 % | 45.1 % |
| `generated_clinical_reasoning` | clinician-style reasoning prompt | 53.6 % | 57.3 % |
| `generated_structured` / `generated_soap` / `generated_plan_focused` / `generated_concise` | task-profile-specific variants | 40-68 % | 40-68 % |

Two important deltas vs v1:

1. **`pair_anchor_b` YES rate dropped 4.7 pp (51.7 % → 47.0 %).** This
   is the v2 library being **less B-overshoot-y** -- the rubrics
   reward thoroughness less reflexively than v1 did.  Combined with
   the +6 pp lift on `pair_anchor_a` recall (per
   `medical_pair_judge_v2_results.md` §4), this is the source of the
   +6.4 pp balanced-accuracy gain.
2. **The new `synthetic` role** (29.2 % YES) gives RL training a
   per-prompt **negative-class** signal that v1 had to fake via
   degraded versions of the anchors.  Useful for contrastive losses or
   for a YES/NO classifier that needs concrete failure cases.

### Aggregate satisfaction rates (full 234,659-row file, sanity check)

The headline rates are computed identically to v1's §3 table.  Run
the snippet in §4 below to recompute on the full file if you need
exact numbers; the first-50k sample above is a fast proxy.

---

## 4. Reading the v2 package

```python
import json

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# 1) Browse rubrics row-by-row
for row in load_jsonl("data/medical_gpt5_b_regen_4k_v2_rubrics.jsonl"):
    print(row["id"], "->", row["rubric_count"], "rubrics")
    for r in row["rubrics"]:
        print(" -", r["source_stage"], r["text"])
    break

# 2) Join slim + rubrics row-for-row (aligned by build order)
slim = load_jsonl("data/medical_gpt5_b_regen_4k_rl.jsonl")     # SAME file as v1
rubr = load_jsonl("data/medical_gpt5_b_regen_4k_v2_rubrics.jsonl")  # v2 stem
for s, r in zip(slim, rubr):
    assert s["id"] == r["id"]
    # train on (s["question"], s["reference_answer_b"], r["rubrics"])
    ...

# 3) Stream evaluations as a reward-model training set
for ev in load_jsonl("data/medical_gpt5_b_regen_4k_v2_rubric_evaluations.jsonl"):
    rubric = ev["rubric_text"]
    response = ev["candidate_text"]
    label = ev["satisfied"]   # bool
    role = ev["candidate_role"]   # 'pair_anchor_a/b' / 'synthetic' / 'generated_*'
    ...

# 4) Stratify by role -- e.g. only pair anchors for a clean A-vs-B classifier
def is_pair_anchor(ev):
    return ev["candidate_role"] in {"pair_anchor_a", "pair_anchor_b"}

pair_only = [ev for ev in load_jsonl("data/medical_gpt5_b_regen_4k_v2_rubric_evaluations.jsonl")
             if is_pair_anchor(ev)]
```

---

## 5. Using the v2 rubrics for RL training

The three v1 recipes (live per-rubric reward; offline reward model;
critique-conditioned SFT/DPO) port to v2 with **only the file stem
changed**.  See `medical_rubrics_4k_package.md` §5.1-5.3 for the full
code.  Quick references:

### 5.1 Recipe A — Live per-rubric reward at rollout time

Identical to v1.  Replace
`data/medical_gpt5_b_regen_4k_rubrics.jsonl` with
`data/medical_gpt5_b_regen_4k_v2_rubrics.jsonl` in the `ROWS` dict
construction; everything else (the `rubric_reward()` function, the
`evaluate_rubric_satisfaction` call, the `JsonlCache`) is unchanged.

The v2 rubrics are slightly more discriminative on average
(post-RRD `discrimination_min_pq=0.05` filter dropped the worst ~10 %),
so the per-rubric YES/NO signal is less noisy than v1's.

### 5.2 Recipe B — Train a reward model offline

Same code as v1 §5.2.  Class balance is now **42.6 % YES / 57.4 % NO**
on the full v2 evaluations file (vs v1's 42.1 / 57.9 %).  Per-role
prior is in §3 above.

**Recommended split:** hold out 10 % of `id`s (not random rows) to
detect rubric-text memorisation.  Stratify by `source` so each held-out
fold contains all five families (`general_instruction_following` is
88 % of rows; `documentation_variants` is only 2 % — a random-row split
under-covers the latter).

### 5.3 Recipe C — Critique-conditioned SFT or DPO from the gold pairs

Same code as v1 §5.3.  Note: the v2 rubrics for the `gold_label`
candidate now satisfy at slightly different rates (47.0 % vs v1's
51.7 % on B; 32.8 % vs 33.3 % on A).  If you build the critique block
by listing only the rubrics that the gold answer satisfies, expect
the chains to be slightly shorter on average than v1.

### 5.4 NEW v2 recipe — Boundary-aware contrastive RL

The v2 `synthetic` candidate role gives a fresh per-prompt
negative-class signal: for every `id`, the file contains evaluations
of a **deliberately terse** candidate (`__boundary_terse`) and a
**deliberately padded-but-uncommitted** candidate
(`__boundary_padded_uncommitted`).  Their YES rate (29.2 %) is
materially below all the LLM-generated candidates (40-67 %), so they
are **clean negatives** for contrastive losses.

```python
def boundary_aware_dpo_pair(row, evals_by_id):
    """Build a DPO pair where 'rejected' is a real-prompt-specific
    boundary candidate, not a generic 'losing' anchor."""
    boundary_terse = next(
        (e["candidate_text"] for e in evals_by_id[row["id"]]
         if e["candidate_id"].endswith("__boundary_terse")),
        None,
    )
    if not boundary_terse:
        return None
    chosen = (
        row["reference_answer_b"] if row["gold_label"] == "b"
        else row["reference_answer_a"]
    )
    return {
        "prompt": row["question"],
        "chosen": chosen,
        "rejected": boundary_terse,    # always a per-prompt weak negative
        "rubrics": [r["text"] for r in row["rubrics"]],
    }
```

This is harder than the v1 §5.3 DPO pair (which used the losing
anchor) -- both `chosen` and `rejected` answer the same question;
`chosen` is medically substantive and `rejected` is technically on
topic but uncommitted/terse.

---

## 6. Caveats specific to the v2 run

These extend the v1 caveats from `medical_rubrics_4k_package.md` §6
with the v2-specific deltas.  The v1 caveats (label prior 73.7 % B,
intra-model regen asymmetry) still apply.

1. **Smaller per-row banks (mean 7.53 vs v1 7.13)** but coming from a
   30 % smaller index (3,944 unique rubrics vs 5,749).  The v2
   index is more concentrated -- top-50 4-token prefixes cover 50.9 %
   of all rubrics (vs 72.2 % in v1), so individual rubrics are less
   templated and the index has higher effective vocabulary.
2. **Score-tie rate ticked up slightly (12.5 % → 14.25 %).** The v2
   strict number is ~1.3 pp lower than v1 strict, but the v2 anti-tie
   cascade resolves 96.5 % of those ties (vs v1 cascade's 42.4 %), so
   the cascade-headline accuracy is +4.4 pp better.
3. **Multi-sample satisfaction (samples=3) was used** for v2's GPT-4o
   judge calls.  The aggregated verdict in `satisfied` is the
   majority vote (yes_votes > no_votes ⇒ True).  Per-sample
   metadata is **not** preserved in this package -- only the final
   verdict.  If you need per-sample variance for confidence-aware
   training, read it directly from the underlying example artifact
   (`methods.rrd_uniform.evaluations[i].metadata.sample_history`).
4. **The `synthetic` role replaces v1's static degradations.**
   In v1 most rows contained no `synthetic` candidates because medical
   Q&A doesn't have a `note_truncated` to degrade.  v2 produces 2
   boundary candidates per row by calling the LLM with deliberately
   weak instructions, so the synthetic role appears on **every** row
   (mean 1.95 synthetic per row vs v1's mean 0.79).
5. **B-overshoot is reduced but not eliminated** (mean
   `sat_a - sat_b = -1.11` vs v1's -1.31).  RL recipes that rely on
   `pair_anchor_b` outscoring `pair_anchor_a` will see ~5 pp lower
   gap than v1 -- which is intentional, but may be worth
   compensating for if your reward function depends on the gap
   magnitude.

---

## 7. Reproduce the package

Deterministic from the existing v2 run directory, ~13 sec:

```bash
python scripts/package_medical_rubrics.py \
    --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k \
    --align-with data/medical_gpt5_b_regen_4k_rl.jsonl \
    --out-dir data \
    --stem medical_gpt5_b_regen_4k_v2
```

Useful flags (same as v1):

- `--no-evaluations` — skip the 402 MB evaluations file when you only
  need the rubric labels.
- `--reasoning-char-limit 1500` — keep more of the per-evaluation
  rationale (default 600).
- `--method rrd_uniform` — use the unweighted method block instead of
  whitened (the raw rubric set is identical, but the embedded
  `ranking` reflects unweighted aggregation).

To regenerate the upstream v2 validation run from scratch (~$700,
~15 hr at `sample_workers=8`), use the v2 runbook in
`medical_rl_prompts_training.md` §"v2 runbook" steps v2-1 → v2-5.
