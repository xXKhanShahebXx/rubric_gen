# Pair-level vs item-level accuracy on RewardBench 2

Quick reference doc explaining the two accuracy columns in
[`reward_bench_2_results.md`](reward_bench_2_results.md).

> TL;DR — they answer two different questions on the same data.
> **Pair-level** counts each (chosen, rejected_i) verdict independently.
> **Item-level** counts an item correct only when the pipeline gets *all three*
> of its pairs right. Item-level is the official RewardBench 2 leaderboard
> metric. Pair-level is a diagnostic that strips out the best-of-4 cubing.

---

## 1. Setup recap

A non-Ties RB2 item is a best-of-4:

```
prompt
chosen completion        (1 item)
rejected completions     (3 per item)
```

Our pipeline can only score things pairwise (A vs B), so we expand each item
into 3 pipeline pairs:

```
RB2 item                                   →    3 pipeline pairs

  prompt:    "important dates in india"
  chosen:    Llama-3.1-8B's correct answer
  rejected:  [Tulu-3-8B's,                  →   pair #1: chosen vs rejected[0]
              Llama-3.1-8B's wrong one,    →   pair #2: chosen vs rejected[1]
              Qwen2.5-7B's]                →   pair #3: chosen vs rejected[2]
```

Each pair becomes an independent JudgeBench-style row: chosen in position A,
rejected in position B, gold label `A>B`. The pipeline outputs a verdict per
pair; the metrics layer regroups them by item.

For 1825 processable RB2 items × 3 = **5,595 pipeline pairs.**

---

## 2. Pair-level — count each verdict independently

> *"Out of the 5,595 pairwise calls the pipeline made, how many got chosen > rejected?"*

The pipeline doesn't even need to know that pair `r0`, `r1`, `r2` came from
the same item. Each pair is just an independent A-vs-B question.

```
pair_acc(subset) = pipeline-correct pairs in subset / total pairs in subset
```

This is the natural pipeline metric — same shape as JudgeBench
single-order accuracy.

---

## 3. Item-level — all three verdicts must be right

> *"Out of the 1825 items, on how many did the pipeline pick chosen over every rejected?"*

A single wrong pair tanks the entire item.

```
item_acc(subset) = items where all 3 pairs are correct / items in subset
```

This is the **official RewardBench 2 leaderboard metric**. It's the
benchmark's definition of "the reward model assigns a higher score to chosen
than to every rejected".

---

## 4. Worked example — 3 items, 9 pairs

```
                  pair r0   pair r1   pair r2
   item p1          ✓         ✓         ✓
   item p2          ✓         ✗         ✓
   item p3          ✗         ✓         ✓

   pair-level = 7/9 = 77.8 %
   item-level = 1/3 = 33.3 %    (only p1 had all 3 right)
```

Note that p2 and p3 both have *2 of 3 correct* and yet count as **wrong** at
item-level. The benchmark only credits "perfect" items.

---

## 5. Same pair-level, very different item-level

The two numbers are not interchangeable. With identical pair-level
accuracy you can land at very different item-level numbers depending on
how the errors are distributed across items.

**Errors spread thin (one wrong per item):**

```
                  r0   r1   r2
   p1              ✓    ✗    ✓
   p2              ✓    ✗    ✓
   p3              ✓    ✗    ✓

   pair-level = 6/9 = 67 %
   item-level = 0/3 = 0 %         ← every item has a wrong pair
```

**Errors concentrated (one item entirely wrong):**

```
                  r0   r1   r2
   p1              ✓    ✓    ✓
   p2              ✓    ✓    ✓
   p3              ✗    ✗    ✗

   pair-level = 6/9 = 67 %        ← same!
   item-level = 2/3 = 67 %        ← much better
```

Same pair-level. Item-level swings from **0 % to 67 %** purely on error
clustering.

---

## 6. The independence floor

If the 3 pairs of an item were truly independent and you got each right
with probability `p`, item-level would be exactly `p³`:

| Pair-level `p` | Independent-errors item-level (`p³`) |
|---:|---:|
| 50 % | 12.5 % |
| 60 % | 21.6 % |
| 70 % | 34.3 % |
| 76 % | 44.0 % |
| 80 % | 51.2 % |
| 90 % | 72.9 % |

Real item-level usually sits *above* the `p³` floor because the 3 pairs of
an item share the same chosen response — when chosen is genuinely strong
the pipeline tends to win all 3, when it's weak it tends to lose all 3.
Errors cluster, which helps item-level.

---

## 7. The actual numbers from `rb2_full_v47`

| Subset | Items | Pair-level acc | Item-level acc | `p³` floor (for reference) |
|---|---:|---:|---:|---:|
| Factuality | 475 | 76.49 % | 53.05 % | 44.7 % |
| Focus | 495 | 74.34 % | 53.33 % | 41.1 % |
| Math | 183 | 83.24 % | 66.67 % | 57.7 % |
| Precise IF | 160 | 58.54 % | 29.38 % | 20.0 % |
| Safety | 450 | 77.78 % | 60.89 % | 47.0 % |
| Ties | 102 | 77.78 % | 62.75 % | 47.0 % |
| **5-subset avg** | 1763 | **74.04 %** | **52.66 %** | 40.5 % |

Each row's actual item-level beats its `p³` floor by 5–13 points, exactly
as expected from the chosen-response clustering effect.

---

## 8. Which one to quote

* **For leaderboard / publication:** quote item-level. That's what the
  RewardBench 2 paper and dashboard report.
* **For pipeline diagnostics:** prefer pair-level. It tells you how good
  the underlying judge is on a single A-vs-B decision, factoring out the
  best-of-4 cubing penalty.
* **For comparing to JudgeBench:** the pair-level number on RB2 is the
  apples-to-apples comparison to JudgeBench v4.7's 80.57 % (which is also
  single-order pairwise). Pair-level on RB2 5-subset avg = 74 %, so the
  rubric-judge quality dropped ~6 points from JudgeBench to RB2 — most
  of that gap is the verifier toolchain not transferring (no MCQ format,
  no LeetCode shape).

The 28-point JudgeBench → RB2 gap (80.57 % → 52.66 %) is therefore really:

```
  ~6 points     rubric judge handles RB2 prompts slightly worse
  ~22 points    best-of-4 cubing penalty (pair-level → item-level)
  -------------
  ~28 points    total
```

Almost all of the apparent regression is the metric shape, not a
regression in the pipeline itself.
