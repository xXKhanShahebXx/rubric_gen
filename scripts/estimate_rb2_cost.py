"""Cost estimate for evaluating the rubric pipeline on allenai/reward-bench-2."""

# Per-pair: rubric discovery + rubric satisfaction (8 criteria × 5 SC × 2 cand) +
# discriminator + self-critique + verifier solvers ~= 85-100 calls
CALLS_PER_PAIR = 90
GPT4O_PER_CALL = 0.006
CLAUDE_PER_CALL = 0.015

NON_TIES_ITEMS = 1763
TIES_ITEMS = 102


def main() -> None:
    non_ties_pairs = NON_TIES_ITEMS * 3
    ties_pairs_capped = TIES_ITEMS * 3
    total_pairs = non_ties_pairs + ties_pairs_capped
    print(f"Total pairs (full, Ties capped to 3 rej): {total_pairs:,}")

    gpt_calls = total_pairs * CALLS_PER_PAIR
    gpt_cost = gpt_calls * GPT4O_PER_CALL
    print(f"GPT-4o calls: {gpt_calls:,}  cost: ${gpt_cost:,.0f}")

    # Claude solver: math solver fires on Math (183 items × 3 pairs = 549 pairs).
    # mmlu_independent_answerer doesn't fire on free-form RB2 prompts (no MCQ letters).
    # reasoning_independent_solver may fire on Precise IF (160 × 3 = 480 pairs).
    claude_pairs = 549 + 480
    claude_calls = claude_pairs * 3
    claude_cost = claude_calls * CLAUDE_PER_CALL
    print(f"Claude solver calls: {claude_calls:,}  cost: ${claude_cost:,.0f}")

    print(f"TOTAL FULL EVAL: ${gpt_cost + claude_cost:,.0f}")
    print()
    print("Smoke options:")
    for n in (2, 5, 10, 25):
        items = n * 6
        pairs = n * 5 * 3 + n * 3
        cost = pairs * CALLS_PER_PAIR * GPT4O_PER_CALL * 1.3
        print(f"  N={n} per subset  items={items:>3} pairs={pairs:>4}  ~${cost:.0f}")


if __name__ == "__main__":
    main()
