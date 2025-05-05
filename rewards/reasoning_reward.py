import re
import time
import math
from typing import List, Set, Dict
from .reward_utils import completions_to_lst, extract_tag_content

# --- Constants ---


'''
修改提示词去激发

(不要分边界)
validate
confirm
verification
verifies
step-by-step
Hence
Given
verified
recalculate
let's
Given
find out
states
which
provided
considering
assume
assuming
assumption
combining
conduct
perform
determine
Breaking down
solve
confirmed
recheck
conclude
calculation
calcula
simplify
'''

# 1. Categorized Keywords and Tier Scores
KEYWORD_CATEGORIES: Dict[str, str] = {
    # Procedural (Low-Mid Score)
    "first": "procedural",
    "firstly": "procedural",
    "to begin with": "procedural",
    "second": "procedural",
    "secondly": "procedural",
    "next": "procedural",
    "then": "procedural",
    "finally": "procedural",
    "in conclusion": "procedural",
    "break this down": "procedural",
    "break it down": "procedural",
    "because": "procedural",
    "since": "procedural",
    "consequently": "procedural",
    "as a result": "procedural",
    "so": "procedural",
    "therefore": "procedural",
    # Reflective (High Score)
    "however": "reflective",
    "earlier": "reflective",
    "but": "reflective",
    "wait": "reflective",
    "reconsider": "reflective",
    "rethink": "reflective",
    "re-evaluate": "reflective",
    "hold on": "reflective",
    "yet": "reflective",
    # Checking (Mid Score)
    "verify": "checking",
    "let's check": "checking",
    "let me check": "checking",
    "examine": "checking",
    "double-check": "checking",
    # Conditional/Exploratory (Mid-High Score)
    "what if": "conditional",
    "perhaps": "conditional",
    "maybe": "conditional",
    "possibly": "conditional",
    "on the other hand": "conditional",
    "alternatively": "conditional",
    "assuming": "conditional",
    "assumption": "conditional",
    "given that": "conditional",
    "another": "conditional",
    # General/Ambiguous (Low Score - Treat like procedural or slightly lower)
    "if": "general",
    "think": "general",
}

TIER_SCORES: Dict[str, float] = {
    "procedural": 0.15,
    "general": 0.10,
    "checking": 0.30,
    "conditional": 0.50,
    "reflective": 0.80,
}

# Ensure all original keywords are categorized (as before)
ALL_KEYWORDS_ORIGINAL: Set[str] = {
    "but",
    "wait",
    "alternatively",
    "earlier",
    "however",
    "so",
    "if",
    "think",
    "therefore",
    "then",
    "reconsider",
    "rethink",
    "verify",
    "yet",
    "re-evaluate",
    "let's check",
    "let me check",
    "first",
    "firstly",
    "to begin with",
    "second",
    "secondly",
    "next",
    "finally",
    "in conclusion",
    "break this down",
    "break it down",
    "because",
    "since",
    "consequently",
    "as a result",
    "assuming",
    "assumption",
    "given that",
    "examine",
    "hold on",
    "double-check",
    "on the other hand",
    "what if",
    "perhaps",
    "maybe",
    "possibly",
    "another",
}
missing_keywords = ALL_KEYWORDS_ORIGINAL - set(KEYWORD_CATEGORIES.keys())
if missing_keywords:
    # print(f"Warning: The following keywords are not categorized: {missing_keywords}") # Optional warning
    for kw in missing_keywords:
        KEYWORD_CATEGORIES[kw] = "general"

# Define required opening phrases for the <think> tag
REQUIRED_OPENINGS: List[str] = [
    "\nOkay, so I need to ",
    "\nOkay, let's ",
    "\nOkay, let me ",
]

# Penalty for incorrect opening if <think> tag exists
OPENING_PENALTY: float = -0.5

# Reasoning Reward Configuration
MAX_REASONING_REWARD: float = 2.5
W_TIERED_SUM: float = 1.0
W_DIVERSITY_COUNT: float = 0.2
W_TOTAL_COUNT: float = 0.05
W_DENSITY: float = 20.0
THRESHOLD_DENSITY: float = 0.20
W_REPETITION: float = 40.0
THRESHOLD_REPETITION_RATIO: float = 0.30
MINIMUM_SCORE_CLAMP: float = -1.0
# ** IMPORTANT: Adjust this scale factor based on observed raw scores **
SCALE_FACTOR: float = 0.3  # Example: if ideal raw score is ~8, 0.3 brings it near 2.5

# --- Pre-computation ---

KEYWORD_PATTERNS_DICT: Dict[str, re.Pattern] = {
    keyword: re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
    for keyword in KEYWORD_CATEGORIES.keys()
}


def reasoning_reward(completions, debug: bool = False, **kwargs) -> List[float]:
    """
    Calculates a reward based on categorized reasoning keywords in <think> tags,
    enforces specific opening phrases, and penalizes keyword stuffing (mainly repetition).
    The reasoning score component aims for a maximum of MAX_REASONING_REWARD (before opening penalty).

    Args:
        completions: The generated text(s) from the model.
        debug (bool, optional): If True, prints detailed calculation steps. Defaults to False.
        **kwargs: Catches any other keyword arguments passed.

    Returns:
        List[float]: A list of reward scores, one for each completion.
    """
    start_time = time.time()

    _completions = completions_to_lst(completions)

    def _score_single_completion(completion: str) -> float:
        """Calculates the reasoning reward for a single completion string."""
        think_content_raw = extract_tag_content(
            completion, "think", strip_content=False
        )

        current_opening_penalty = 0.0
        raw_reasoning_score = 0.0

        if think_content_raw is not None:
            if debug:
                print(f"\nProcessing Completion...")
            if debug:
                print(f"  Found <think> tag.")
            # 1. Check for mandatory opening phrase
            opening_ok = False
            for opening in REQUIRED_OPENINGS:
                if think_content_raw.startswith(opening):
                    opening_ok = True
                    if debug:
                        print(f"    Opening '{opening[:15]}...' OK.")
                    break
            if not opening_ok:
                current_opening_penalty = OPENING_PENALTY
                if debug:
                    print(
                        f"    Opening phrase check FAILED. Applying penalty: {OPENING_PENALTY}"
                    )

            # 2. Calculate Keyword Rewards and Penalties
            think_content_processed = think_content_raw.strip()
            think_content_lower = think_content_processed.lower()
            if not think_content_lower:
                if debug:
                    print("    <think> tag content is empty after stripping.")
                final_score = max(MINIMUM_SCORE_CLAMP, current_opening_penalty)
                return final_score

            unique_keywords_found: Set[str] = set()
            keyword_counts: dict[str, int] = {}
            total_keyword_count: int = 0

            # Find all keyword occurrences and categorize
            if debug:
                print("    Scanning for keywords...")
            for keyword, pattern in KEYWORD_PATTERNS_DICT.items():
                matches = pattern.findall(think_content_lower)
                count = len(matches)
                if count > 0:
                    unique_keywords_found.add(keyword)
                    keyword_counts[keyword] = count
                    total_keyword_count += count
                    if debug:
                        category = KEYWORD_CATEGORIES.get(keyword, "unknown")
                        print(
                            f"      Found '{keyword}' (Category: {category}) x {count}"
                        )

            unique_keyword_count = len(unique_keywords_found)
            if debug:
                print(
                    f"    Found {unique_keyword_count} unique keywords, total count {total_keyword_count}."
                )

            # Calculate Positive Reward Component
            sum_tiered_scores = 0.0
            for unique_kw in unique_keywords_found:
                category = KEYWORD_CATEGORIES.get(unique_kw, "general")
                score = TIER_SCORES.get(category, 0.0)
                sum_tiered_scores += score
            if debug:
                print(
                    f"    Sum of Tiered Scores for Unique Keywords: {sum_tiered_scores:.3f}"
                )

            reward_tiered = W_TIERED_SUM * sum_tiered_scores
            reward_diversity = W_DIVERSITY_COUNT * math.log(1 + unique_keyword_count)
            reward_total = W_TOTAL_COUNT * math.log(1 + total_keyword_count)
            r_keywords_positive = reward_tiered + reward_diversity + reward_total
            if debug:
                print(
                    f"    Positive reward components: Tiered={reward_tiered:.3f}, Diversity={reward_diversity:.3f}, Total={reward_total:.3f}"
                )
                print(f"    -> Total Positive Raw Reward = {r_keywords_positive:.3f}")

            # Calculate Negative Penalty Component
            words = think_content_processed.split()
            total_word_count = len(words)
            penalty_density = 0.0
            penalty_repetition = 0.0

            if total_word_count > 0:
                # Density Penalty
                density = total_keyword_count / total_word_count
                penalty_density = W_DENSITY * max(0, density - THRESHOLD_DENSITY) ** 2
                if debug:
                    print(
                        f"    Keyword density: {density:.3f} (Threshold: {THRESHOLD_DENSITY}) -> Density Penalty={penalty_density:.3f}"
                    )

                # Repetition Penalty
                if total_keyword_count > 0:
                    max_single_keyword_count = 0
                    most_repeated_kw = "N/A"
                    if keyword_counts:
                        try:  # Added try-except for safety if keyword_counts is empty after all
                            most_repeated_kw, max_single_keyword_count = max(
                                keyword_counts.items(), key=lambda item: item[1]
                            )
                        except ValueError:
                            pass  # Keep max_single_keyword_count = 0

                    max_single_keyword_freq = (
                        max_single_keyword_count / total_keyword_count
                        if total_keyword_count > 0
                        else 0
                    )
                    penalty_repetition = (
                        W_REPETITION
                        * max(0, max_single_keyword_freq - THRESHOLD_REPETITION_RATIO)
                        ** 2
                    )
                    if debug:
                        print(
                            f"    Max keyword freq: '{most_repeated_kw}' @ {max_single_keyword_freq:.3f} (Threshold: {THRESHOLD_REPETITION_RATIO}) -> Repetition Penalty={penalty_repetition:.3f}"
                        )

            p_stuffing_negative = penalty_density + penalty_repetition
            if debug:
                print(f"    Total stuffing penalty: {p_stuffing_negative:.3f}")

            # Calculate Raw Reasoning Score
            raw_reasoning_score = r_keywords_positive - p_stuffing_negative
            if debug:
                print(
                    f"    Raw Reasoning Score (Positive - Penalties): {raw_reasoning_score:.3f}"
                )

            # Scale and Clamp
            scaled_reasoning_score = raw_reasoning_score * SCALE_FACTOR
            clamped_reasoning_score = max(
                0, min(scaled_reasoning_score, MAX_REASONING_REWARD)
            )
            if debug:
                print(
                    f"    Scaled Reasoning Score (Factor {SCALE_FACTOR:.2f}): {scaled_reasoning_score:.3f}"
                )
                print(
                    f"    Clamped Reasoning Score [0, {MAX_REASONING_REWARD}]: {clamped_reasoning_score:.3f}"
                )

            # Final score calculation
            final_score = clamped_reasoning_score + current_opening_penalty
            final_score = max(MINIMUM_SCORE_CLAMP, final_score)
            if debug:
                print(
                    f"    Final Score (Clamped Reasoning + Opening Penalty, Min Clamp {MINIMUM_SCORE_CLAMP}): {final_score:.3f}"
                )
            return final_score

        else:  # <think> tag does not exist
            # No debug print here as it can be very common
            return 0.0

    # Calculate rewards for all completions
    reasoning_rewards = [_score_single_completion(c) for c in _completions]

    # Print summary only if debugging is enabled
    if debug:
        print(f"\nBatch Reasoning Rewards Calculation Summary:")
        print(
            f"  Target Max Reasoning Reward (after scaling, before penalty): {MAX_REASONING_REWARD}"
        )
        print(f"  Opening Penalty: {OPENING_PENALTY}")

    print(
        f"Reasoning Rewards: {[round(score, 3) for score in reasoning_rewards]} ({time.time() - start_time:.3f} s)"
    )
    print("\n" + "=" * 90 + "\n")

    return reasoning_rewards


# --- Example Usage (for testing with debug enabled) ---
if __name__ == "__main__":
    # Example completions (same as before)
    completion_ok = """
Some introduction.
<think>
Okay, let's break this down. First, I need to identify the core question. Second, I should check the assumptions. However, the data seems sparse. Therefore, maybe I should reconsider the approach. What if I assume X? Let me check the source again. Okay, given that, perhaps the answer is Y. Alternatively, it could be Z. Let's double-check. Finally, I think Y is more likely because... hold on, is assumption X valid? Yes, earlier analysis supports it. But wait, what about edge case Z? Re-evaluate based on Z.
</think>
The final answer is Y.
"""

    completion_bad_opening = """
Some introduction.
<think>
Hmm, I need to figure this out. First, let's think about the premises. However, there's a contradiction. So, therefore, the conclusion must be wrong. Let me check. Yes, I should re-evaluate.
</think>
The final answer is invalid.
"""

    completion_no_think = """
This is a direct answer without a think block.
The answer is probably X.
"""

    completion_stuffing_repetitive = """
Some text.
<think>
Okay, let's think think think think think think think think. Maybe maybe maybe maybe maybe maybe maybe. Because because because because. Think think think think. Maybe think? Let me check, maybe think. Think!
</think>
Answer: think.
"""

    completion_stuffing_dense_varied = """
Some text.
<think>
Okay, let's examine the problem. First, however, maybe check assumptions. But wait, reconsider this. What if perhaps alternatively... let me check. Secondly, verify the source. Therefore, hold on, re-evaluate given that earlier assumption. Possibly examine... think! Since this... then... finally...
</think>
Answer: TBD.
"""

    completion_empty_think = """
Intro.
<think>
</think>
Conclusion.
"""

    completion_good_short_procedural = """
Blah blah.
<think>
Okay, let me check. First, examine the input. Since it's positive, then the result should be positive too. Finally, return positive.
</think>
Result: Positive.
"""

    completion_good_short_reflective = """
Blah blah.
<think>
Okay, let's reconsider. However, what if the input was negative? Wait, let me check the requirements again. Perhaps that case is excluded.
</think>
Result: Positive (assuming positive input).
"""

    test_completions = [
        completion_ok,
        completion_bad_opening,
        completion_no_think,
        completion_stuffing_repetitive,
        completion_stuffing_dense_varied,
        completion_empty_think,
        completion_good_short_procedural,
        completion_good_short_reflective,
    ]

    print("--- Running reasoning_reward with debug=True ---")
    # Call with debug=True to see detailed output
    rewards_debug = reasoning_reward(test_completions, debug=True)

    print("\n--- Running reasoning_reward with debug=False (Default) ---")
    # Call without debug argument (defaults to False) - should be quiet
    rewards_no_debug = reasoning_reward(test_completions)
    print(f"Rewards (debug=False): {[round(r, 3) for r in rewards_no_debug]}")
