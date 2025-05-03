import re, time
from typing import List, Dict, Tuple, Any
from rewards.reward_utils import completions_to_lst

# --- Constants ---
TAG_THINK_START = "<think>"
TAG_THINK_END = "</think>"
TAG_ANSWER_START = "<answer>"
TAG_ANSWER_END = "</answer>"
REQUIRED_TAGS = [TAG_THINK_START, TAG_THINK_END, TAG_ANSWER_START, TAG_ANSWER_END]

# Stricter perfect patterns (matched against the raw response string)
PERFECT_PATTERN_NO_FINAL_NL = re.compile(
    r"^<think>\n\s*[^\s][\s\S]*?\n</think>\n<answer>\n\s*[^\s][\s\S]*?\n</answer>$"
)
PERFECT_PATTERN_WITH_FINAL_NL = re.compile(  # Kept for checking if it was matched, but won't give perfect bonus
    r"^<think>\n\s*[^\s][\s\S]*?\n</think>\n<answer>\n\s*[^\s][\s\S]*?\n</answer>\n$"
)

# LaTeX Patterns (matched within the extracted answer content string)
LATEX_BOXED_PATTERN = re.compile(
    r"\\boxed{.+?}"
)  # Non-greedy match for content inside \boxed{}
# Removed unused regex patterns for newlines


# --- Rewards ---
R_TAG_EXISTS_ONCE = 0.10  # Max 0.40
R_CORRECT_PAIR_ORDER = 0.05  # Max 0.10
R_CORRECT_BLOCK_ORDER = 0.05
R_NEWLINE_AFTER_OPEN = 0.05  # Max 0.10
R_NEWLINE_BEFORE_CLOSE = 0.05  # Max 0.10
R_NEWLINE_BETWEEN_BLOCKS = 0.10
R_PERFECT_MATCH_EXTERNAL = 0.20  # ONLY for NO_FINAL_NL pattern AND ending

# LaTeX specific rewards (within <answer> block content)
R_LATEX_DOLLARS_PRESENT = 0.10  # Presence of at least two $$
R_LATEX_BOXED_PRESENT = 0.10  # Presence of \boxed{...} within $$...$$
R_LATEX_NEWLINE_BEFORE_BOXED = 0.05  # Presence of \n *immediately* after first $$
R_LATEX_NEWLINE_AFTER_BOXED = 0.05  # Presence of \n *immediately* before last $$
R_LATEX_PERFECT_BLOCK = (
    0.20  # Bonus triggered if all 4 above LaTeX components are present.
)


# --- Penalties ---
P_TAG_DUPLICATED = -0.20
P_CONTENT_OUTSIDE_BASE = -0.10
P_CONTENT_OUTSIDE_PER_CHAR = -0.005
P_EMPTY_CONTENT = -0.20
P_ORDER_VIOLATION = -0.15


# --- Reward Limits ---
MIN_REWARD_FLOOR = -1.5


# === Helper Functions ===


def _find_tags(response: str) -> Dict[str, int]:
    """Finds the first occurrence position of each required tag."""
    return {tag: response.find(tag) for tag in REQUIRED_TAGS}


def _count_tags(response: str) -> Dict[str, int]:
    """Counts occurrences of each required tag."""
    return {tag: response.count(tag) for tag in REQUIRED_TAGS}


def _check_tag_existence_and_count(
    tag_counts: Dict[str, int], debug_scores: Dict[str, Any]
) -> float:
    """Calculates reward/penalty based on tag counts."""
    reward = 0.0
    all_present_once = True
    step_scores = {}
    for tag in REQUIRED_TAGS:
        count = tag_counts.get(tag, 0)
        if count == 1:
            step_score = R_TAG_EXISTS_ONCE
            reward += step_score
            step_scores[f"{tag}_exists_once"] = step_score
        else:
            all_present_once = False
            if count > 1:
                step_score = P_TAG_DUPLICATED * (count - 1)
                reward += step_score
                step_scores[f"{tag}_duplicated_x{count-1}"] = step_score
    debug_scores["tag_existence_details"] = step_scores
    return reward


def _check_tag_order(
    tag_positions: Dict[str, int],
    tag_counts: Dict[str, int],
    debug_scores: Dict[str, Any],
) -> float:
    """Calculates reward/penalty based on tag order."""
    reward = 0.0
    step_scores = {}
    think_start_pos, think_end_pos = (
        tag_positions.get(TAG_THINK_START, -1),
        tag_positions.get(TAG_THINK_END, -1),
    )
    answer_start_pos, answer_end_pos = (
        tag_positions.get(TAG_ANSWER_START, -1),
        tag_positions.get(TAG_ANSWER_END, -1),
    )

    has_think_pair = (
        tag_counts.get(TAG_THINK_START, 0) == 1
        and tag_counts.get(TAG_THINK_END, 0) == 1
    )
    has_answer_pair = (
        tag_counts.get(TAG_ANSWER_START, 0) == 1
        and tag_counts.get(TAG_ANSWER_END, 0) == 1
    )

    think_order_correct = (
        has_think_pair
        and think_start_pos != -1
        and think_end_pos != -1
        and think_start_pos < think_end_pos
    )
    answer_order_correct = (
        has_answer_pair
        and answer_start_pos != -1
        and answer_end_pos != -1
        and answer_start_pos < answer_end_pos
    )

    if think_order_correct:
        step_scores["think_pair_order"] = R_CORRECT_PAIR_ORDER
        reward += R_CORRECT_PAIR_ORDER
    elif has_think_pair:
        step_scores["think_pair_order_violation"] = P_ORDER_VIOLATION
        reward += P_ORDER_VIOLATION

    if answer_order_correct:
        step_scores["answer_pair_order"] = R_CORRECT_PAIR_ORDER
        reward += R_CORRECT_PAIR_ORDER
    elif has_answer_pair:
        step_scores["answer_pair_order_violation"] = P_ORDER_VIOLATION
        reward += P_ORDER_VIOLATION

    if think_order_correct and answer_order_correct:
        if think_end_pos < answer_start_pos:
            step_scores["block_order"] = R_CORRECT_BLOCK_ORDER
            reward += R_CORRECT_BLOCK_ORDER
        else:
            step_scores["block_order_violation"] = P_ORDER_VIOLATION
            reward += P_ORDER_VIOLATION

    debug_scores["tag_order_details"] = step_scores
    return reward


def _penalize_extraneous_content(
    response: str,
    tag_positions: Dict[str, int],
    tag_counts: Dict[str, int],
    debug_scores: Dict[str, Any],
) -> float:
    """Calculates penalties for content outside the main tags or non-whitespace between blocks."""
    penalty = 0.0
    step_scores = {}
    think_start_pos, think_end_pos = (
        tag_positions.get(TAG_THINK_START, -1),
        tag_positions.get(TAG_THINK_END, -1),
    )
    answer_start_pos, answer_end_pos = (
        tag_positions.get(TAG_ANSWER_START, -1),
        tag_positions.get(TAG_ANSWER_END, -1),
    )

    has_think_start = tag_counts.get(TAG_THINK_START, 0) > 0 and think_start_pos != -1
    has_answer_end = tag_counts.get(TAG_ANSWER_END, 0) > 0 and answer_end_pos != -1

    if has_think_start and think_start_pos > 0:
        before_think = response[:think_start_pos].strip()
        if before_think:
            score = (
                P_CONTENT_OUTSIDE_BASE + len(before_think) * P_CONTENT_OUTSIDE_PER_CHAR
            )
            penalty += score
            step_scores["before_think_penalty"] = score

    if has_answer_end:
        after_answer_start_idx = answer_end_pos + len(TAG_ANSWER_END)
        if after_answer_start_idx < len(response):
            after_answer = response[
                after_answer_start_idx:
            ].strip()  # Check for non-whitespace after
            if after_answer:
                score = (
                    P_CONTENT_OUTSIDE_BASE
                    + len(after_answer) * P_CONTENT_OUTSIDE_PER_CHAR
                )
                penalty += score
                step_scores["after_answer_penalty"] = score

    think_pair_valid = (
        tag_counts.get(TAG_THINK_START, 0) == 1
        and tag_counts.get(TAG_THINK_END, 0) == 1
        and think_start_pos != -1
        and think_end_pos != -1
        and think_start_pos < think_end_pos
    )
    answer_pair_valid = (
        tag_counts.get(TAG_ANSWER_START, 0) == 1
        and tag_counts.get(TAG_ANSWER_END, 0) == 1
        and answer_start_pos != -1
        and answer_end_pos != -1
        and answer_start_pos < answer_end_pos
    )
    block_order_valid = (
        think_pair_valid and answer_pair_valid and think_end_pos < answer_start_pos
    )

    if block_order_valid:
        between_start_idx = think_end_pos + len(TAG_THINK_END)
        between_content = response[between_start_idx:answer_start_pos]
        between_content_stripped = between_content.strip()

        if between_content_stripped != "":
            score = (
                P_CONTENT_OUTSIDE_BASE
                + len(between_content_stripped) * P_CONTENT_OUTSIDE_PER_CHAR
            )
            penalty += score
            step_scores["between_blocks_penalty"] = score

    debug_scores["extraneous_content_details"] = step_scores
    return penalty


def _check_internal_formatting(
    response: str,
    tag_positions: Dict[str, int],
    tag_counts: Dict[str, int],
    debug_scores: Dict[str, Any],
) -> float:
    """Checks specific newlines around open/close tags and between blocks."""
    reward = 0.0
    step_scores = {}
    think_start_pos, think_end_pos = (
        tag_positions.get(TAG_THINK_START, -1),
        tag_positions.get(TAG_THINK_END, -1),
    )
    answer_start_pos, answer_end_pos = (
        tag_positions.get(TAG_ANSWER_START, -1),
        tag_positions.get(TAG_ANSWER_END, -1),
    )

    think_pair_valid_for_nl = (
        tag_counts.get(TAG_THINK_START, 0) == 1
        and tag_counts.get(TAG_THINK_END, 0) == 1
        and think_start_pos != -1
        and think_end_pos != -1
        and think_start_pos < think_end_pos
    )
    answer_pair_valid_for_nl = (
        tag_counts.get(TAG_ANSWER_START, 0) == 1
        and tag_counts.get(TAG_ANSWER_END, 0) == 1
        and answer_start_pos != -1
        and answer_end_pos != -1
        and answer_start_pos < answer_end_pos
    )
    block_order_valid_for_nl = (
        think_pair_valid_for_nl
        and answer_pair_valid_for_nl
        and think_end_pos < answer_start_pos
    )

    if think_pair_valid_for_nl:
        content_start = think_start_pos + len(TAG_THINK_START)
        if content_start < think_end_pos and response[content_start] == "\n":
            step_scores["think_after_open"] = R_NEWLINE_AFTER_OPEN
            reward += R_NEWLINE_AFTER_OPEN
        tag_end_start_idx = think_end_pos
        if (
            think_end_pos > content_start
            and tag_end_start_idx > 0
            and response[tag_end_start_idx - 1] == "\n"
        ):
            step_scores["think_before_close"] = R_NEWLINE_BEFORE_CLOSE
            reward += R_NEWLINE_BEFORE_CLOSE

    if answer_pair_valid_for_nl:
        content_start = answer_start_pos + len(TAG_ANSWER_START)
        if content_start < answer_end_pos and response[content_start] == "\n":
            step_scores["answer_after_open"] = R_NEWLINE_AFTER_OPEN
            reward += R_NEWLINE_AFTER_OPEN
        tag_end_start_idx = answer_end_pos
        if (
            answer_end_pos > content_start
            and tag_end_start_idx > 0
            and response[tag_end_start_idx - 1] == "\n"
        ):
            step_scores["answer_before_close"] = R_NEWLINE_BEFORE_CLOSE
            reward += R_NEWLINE_BEFORE_CLOSE

    if block_order_valid_for_nl:
        between_start_idx = think_end_pos + len(TAG_THINK_END)
        between_content = response[between_start_idx:answer_start_pos]
        if between_content.strip() == "" and "\n" in between_content:
            step_scores["between_blocks_newline"] = R_NEWLINE_BETWEEN_BLOCKS
            reward += R_NEWLINE_BETWEEN_BLOCKS

    debug_scores["internal_formatting_details"] = step_scores
    return reward


def _check_content_and_latex(
    think_content: str, answer_content: str, debug_scores: Dict[str, Any]
) -> float:
    """Checks for content emptiness and LaTeX format within answer."""
    reward = 0.0
    step_scores: Dict[str, float | bool | str | None] = {}

    if not think_content.strip():
        step_scores["think_empty_penalty"] = P_EMPTY_CONTENT
        reward += P_EMPTY_CONTENT
    else:
        step_scores["think_not_empty"] = 0.0

    answer_stripped = answer_content.strip()
    has_dollars = False
    has_boxed = False
    has_newline_after_dollars = False
    has_newline_before_dollars = False
    contains_required_latex_structure = False

    dollar_matches = list(re.finditer(r"\$\$", answer_content))

    if len(dollar_matches) >= 2:
        step_scores["latex_dollars"] = R_LATEX_DOLLARS_PRESENT  # Assign potential score
        has_dollars = True

        first_dollar_end_idx = dollar_matches[0].span()[1]
        last_dollar_start_idx = dollar_matches[-1].span()[0]

        if first_dollar_end_idx < last_dollar_start_idx:
            latex_block_potential_content = answer_content[
                first_dollar_end_idx:last_dollar_start_idx
            ]
            if re.search(LATEX_BOXED_PATTERN, latex_block_potential_content):
                step_scores["latex_boxed"] = (
                    R_LATEX_BOXED_PRESENT  # Assign potential score
                )
                has_boxed = True
        elif has_dollars:
            step_scores["latex_boxed"] = 0.0

        # FIX: Use ONLY precise index check for awarding newline scores
        if (
            first_dollar_end_idx < len(answer_content)
            and answer_content[first_dollar_end_idx] == "\n"
        ):
            step_scores["latex_dollars_newline_after"] = (
                R_LATEX_NEWLINE_BEFORE_BOXED  # Assign score for $$\n
            )
            has_newline_after_dollars = True

        if (
            last_dollar_start_idx > 0
            and answer_content[last_dollar_start_idx - 1] == "\n"
        ):
            step_scores["latex_newline_dollars_before"] = (
                R_LATEX_NEWLINE_AFTER_BOXED  # Assign score for \n$$
            )
            has_newline_before_dollars = True

        # Debug flags
        step_scores["debug_has_dollars"] = has_dollars
        step_scores["debug_has_boxed"] = has_boxed
        step_scores["debug_has_nl_after_dollars"] = has_newline_after_dollars
        step_scores["debug_has_nl_before_dollars"] = has_newline_before_dollars

        if (
            has_dollars
            and has_boxed
            and has_newline_after_dollars
            and has_newline_before_dollars
        ):
            step_scores["latex_perfect_block_bonus"] = (
                R_LATEX_PERFECT_BLOCK  # Assign potential score
            )
            contains_required_latex_structure = True

    # Penalty for empty answer content...
    if not answer_stripped and not contains_required_latex_structure:
        step_scores["answer_empty_penalty"] = P_EMPTY_CONTENT
        reward += P_EMPTY_CONTENT
    else:
        step_scores["answer_not_empty"] = 0.0

    # Add up the LaTeX rewards found (excluding debug flags)
    latex_reward_keys = [
        "latex_dollars",
        "latex_boxed",
        "latex_dollars_newline_after",
        "latex_newline_dollars_before",
        "latex_perfect_block_bonus",
    ]
    latex_reward_sum = sum(
        step_scores.get(k, 0.0)
        for k in latex_reward_keys
        if isinstance(step_scores.get(k), (int, float))
    )

    reward += latex_reward_sum

    debug_scores["content_latex_details"] = step_scores
    return reward


def _check_perfect_match_external(response: str, debug_scores: Dict[str, Any]) -> float:
    """Checks if the response perfectly matches the strict target structure (no trailing NL)."""
    score = 0.0
    step_scores: Dict[str, float | str | bool] = {}

    matched_no_nl_pattern = PERFECT_PATTERN_NO_FINAL_NL.match(response)
    matched_with_nl_pattern = PERFECT_PATTERN_WITH_FINAL_NL.match(response)
    ends_with_tag_end = response.endswith(TAG_ANSWER_END)

    # Debug flags for pattern matching and ending
    step_scores["matched_no_nl_pattern"] = bool(matched_no_nl_pattern)
    step_scores["matched_with_nl_pattern"] = bool(matched_with_nl_pattern)
    step_scores["ends_with_tag_end"] = ends_with_tag_end

    # FIX: Award R_PERFECT_MATCH_EXTERNAL ONLY for the pattern that implies NO final NL,
    # AND verify the string actually ends exactly with </answer>.
    if matched_no_nl_pattern and ends_with_tag_end:
        # This is the strict target format for the bonus
        score = R_PERFECT_MATCH_EXTERNAL
        step_scores["perfect_match_no_final_nl_bonus_awarded"] = score
    elif matched_with_nl_pattern:
        # Matched the pattern including a final newline - not the strict target for bonus
        step_scores["perfect_match_with_final_nl_no_bonus"] = True
        score = 0.0
    elif matched_no_nl_pattern and not ends_with_tag_end:
        # Matched the NO_NL pattern but the string actually ended in \n (due to $ behavior).
        # Not the strict target for bonus.
        step_scores["perfect_match_no_nl_pattern_ends_nl_no_bonus"] = True
        score = 0.0
    else:
        # Didn't match either main pattern
        step_scores["perfect_match_none"] = True
        score = 0.0

    debug_scores["perfect_external_match"] = step_scores
    return score


def format_reward(completions, debug=False, **kwargs) -> List[float]:
    """
    Calculates a refined reward score for completions based on adherence
    to the <think>...</think>\n<answer>...</answer> format, with stricter checks,
    including specific rewards for LaTeX formatting within the <answer> block.

    Args:
        completions: Input completions, processed by completions_to_lst.
        debug (bool): If True, prints detailed scores for each completion.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        A list of float reward scores.
    """
    start_time = time.time()

    _completions = completions_to_lst(completions)

    final_rewards = []
    for i, response in enumerate(_completions):
        total_reward = 0.0
        debug_scores: Dict[str, Any] = {}  # Dictionary to store detailed scores

        if debug:
            print(f"\n--- Processing Completion {i} ---")
            debug_scores["raw_response"] = repr(response)

        # 1. Tag Counts and Positions
        tag_counts = _count_tags(response)
        tag_positions = _find_tags(response)
        if debug:
            debug_scores["tag_counts"] = tag_counts
            debug_scores["tag_positions"] = tag_positions

        # 2. Tag Existence
        total_reward += _check_tag_existence_and_count(tag_counts, debug_scores)

        # Extract indices for structure checks
        think_start_pos = tag_positions.get(TAG_THINK_START, -1)
        think_end_pos = tag_positions.get(TAG_THINK_END, -1)
        answer_start_pos = tag_positions.get(TAG_ANSWER_START, -1)
        answer_end_pos = tag_positions.get(TAG_ANSWER_END, -1)

        # 3. Tag Order
        total_reward += _check_tag_order(tag_positions, tag_counts, debug_scores)

        # Determine if basic structure allows content checks
        think_pair_valid_for_content = (
            tag_counts.get(TAG_THINK_START, 0) == 1
            and tag_counts.get(TAG_THINK_END, 0) == 1
            and think_start_pos != -1
            and think_end_pos != -1
            and think_start_pos < think_end_pos
        )
        answer_pair_valid_for_content = (
            tag_counts.get(TAG_ANSWER_START, 0) == 1
            and tag_counts.get(TAG_ANSWER_END, 0) == 1
            and answer_start_pos != -1
            and answer_end_pos != -1
            and answer_start_pos < answer_end_pos
        )
        block_order_valid_for_content = (
            think_pair_valid_for_content
            and answer_pair_valid_for_content
            and think_end_pos < answer_start_pos
        )

        if block_order_valid_for_content:
            think_content_start_idx = think_start_pos + len(TAG_THINK_START)
            think_content = response[think_content_start_idx:think_end_pos]
            answer_content_start_idx = answer_start_pos + len(TAG_ANSWER_START)
            answer_content = response[answer_content_start_idx:answer_end_pos]

            if debug:
                debug_scores["extracted_think_content"] = repr(think_content)
                debug_scores["extracted_answer_content"] = repr(answer_content)

            # 4. Extraneous Content
            total_reward += _penalize_extraneous_content(
                response, tag_positions, tag_counts, debug_scores
            )
            # 5. Internal Formatting (Newlines)
            total_reward += _check_internal_formatting(
                response, tag_positions, tag_counts, debug_scores
            )
            # 6. Content Emptiness & LaTeX
            total_reward += _check_content_and_latex(
                think_content, answer_content, debug_scores
            )
        else:
            if debug:
                debug_scores["content_internal_checks_skipped"] = True

        # 7. Perfect External Match
        total_reward += _check_perfect_match_external(response, debug_scores)

        # 8. Final Score
        final_score = max(MIN_REWARD_FLOOR, total_reward)
        debug_scores["total_before_floor"] = total_reward
        debug_scores["final_score"] = final_score
        final_rewards.append(final_score)

        if debug:
            print(
                f"\n--- Debug Scores Summary for Completion {i} (Total: {round(final_score, 4)}) ---"
            )
            verbose_keys_to_print = [
                "raw_response",
                "extracted_think_content",
                "extracted_answer_content",
                "tag_counts",
                "tag_positions",
                "content_internal_checks_skipped",
                "total_before_floor",
                "final_score",
            ]
            for key in verbose_keys_to_print:
                if key in debug_scores:
                    print(f"  {key}: {debug_scores[key]}")

            category_keys = [
                "tag_existence_details",
                "tag_order_details",
                "extraneous_content_details",
                "internal_formatting_details",
                "content_latex_details",
                "perfect_external_match",
            ]
            for key in sorted(category_keys):
                if key in debug_scores:
                    value = debug_scores[key]
                    if isinstance(value, dict):
                        total_in_category = sum(
                            v for v in value.values() if isinstance(v, (int, float))
                        )
                        print(f"  {key}: {round(total_in_category, 4)}")
                        if value:
                            print(f"    Details: {value}")

            print("--------------------------------------")

    print(
        f"Format Rewards: {[round(score, 3) for i, score in enumerate(final_rewards)]} ({time.time() - start_time:.3f} s)"
    )
    return final_rewards


if __name__ == "__main__":
    # --- Example Usage ---
    completions_data_list = [
        "<think>\n  Thinking process here.\n</think>\n<answer>\n  Final answer here.\n</answer>",  # 0: NO final NL perfect external. Expected: 1.05.
        "Oops<think>\nThink\n</think>\n<answer>Ans</answer>",  # 1: Junk before, missing NLs, no LaTeX. Expected: 0.63.
        "<think>\n </think>\n<answer>\n </answer>",  # 2: Empty content, NO final NL. Expected: 0.35.
        "<think>T</think><answer>A</answer>",  # 3: No NLs, NO final NL. Expected: 0.55.
        "<think>\n T \n</think> \n <answer>\n A \n</answer>\n",  # 4: WITH final NL, Extra space. Expected: 0.85.
        "<think>\n T \n</think> EXTRA JUNK <answer>\n A \n</answer>\n",  # 5: Junk between, WITH final NL. Expected: 0.6.
        "<think>\n T \n</think>\n<answer>\n A \n</answer>EXTRA AFTER",  # 6: Junk after, NO final NL. Expected: 0.695.
        "<think>\n T \n</think>\n<answer>\n A \n</answer>\n",  # 7: WITH final NL. Expected: 0.85.
        "<think>\nT\n</think>\n<answer>\nA\n</answer>",  # 8: NO final NL perfect external. Expected: 1.05.
        "<think>\n T \n</think>\n<answer>\n A \n</answer>\n\n",  # 9: Extra final NL. Expected: 0.85.
        "<think>\nGood\n</think>\n<answer>\nOkay\n</answer>",  # 10: NO final NL perfect external. Expected: 1.05.
        "Bad format",  # 11: Expected: 0.0.
        "<think>\nOnly think tag.</think>",  # 12: Missing tags. Expected: 0.25.
        "<answer>\nOnly answer tag.\n</answer>",  # 13: Missing tags. Expected: 0.25.
        "<think>\nThink</answer>\n<answer>\nAnswer</think>",  # 14: Wrong order. Expected: 0.3.
        # --- Examples Testing LaTeX Requirements ---
        # Expected scores recalculated assuming Strict External Perfect Bonus (ONLY NO_FINAL_NL)
        "<think>\nThis is thinking.\n</think>\n<answer>\n$$\n\\boxed{The Final Answer}\n$$\n</answer>\n",  # 15: WITH final NL. Perfect LaTeX. Expected: 0.4+0.15+0+0.3+0(Ext)+0.5(LaTeX)=1.35.
        "<think>\nThinking...\n</think>\n<answer>\n$$ \\boxed{Answer} $$\n</answer>",  # 16: NO final NL perfect external. Basic LaTeX (no internal NLs). Expected: 0.4+0.15+0+0.3+0.2(Ext)+0.2(LaTeX)=1.25.
        "<think>\nThinking...\n</think>\n<answer>\nThe answer is \\boxed{Result}.\n</answer>",  # 17: NO final NL perfect external. No LaTeX. Expected: 0.4+0.15+0+0.3+0.2(Ext)+0=1.05.
        "<think>\nThinking...\n</think>\n<answer>\nMy attempt: $$ formula $$\n</answer>",  # 18: NO final NL perfect external. $$ only. Expected: 0.4+0.15+0+0.3+0.2(Ext)+0.1(LaTeX)=1.15.
        "<think>\nThinking...\n</think>\n<answer>\n   \n \n </answer>",  # 19: NO final NL perfect external, Empty answer. Expected: 0.4+0.15+0+0.3+0.2(Ext)-0.2(Empty)=0.85.
        "<think>\n   \n \n </think>\n<answer>\nAnswer is here.\n</answer>",  # 20: NO final NL perfect external, Empty think. Expected: 0.4+0.15+0+0.3+0.2(Ext)-0.2(Empty)=0.85.
        "<think>\nOkay.\n</think>\n<answer>\n$$\n\\boxed{Result}\n$$",  # 21: Missing </answer> tag. Expected: 0.35.
        "<think>\nOkay.\n</think>\n<answer>$$\n\\boxed{Result}\n$$</answer>",  # 22: NO final NL. Missing answer_after_open NL. Perfect LaTeX. Expected: 0.4+0.15+0+0.2(Internal)+0(Ext)+0.5(LaTeX)=1.25.
        "<think>\nOkay.\n</think>\n<answer>\nSome text before. $$<CR>\n\\boxed{Result}<CR>\n$$<CR>\nSome text after.<CR>\n</answer>\n".replace(
            "<CR>", "\n"
        ),  # 23: WITH final NL. Perfect LaTeX + other text. Expected: 0.4+0.15+0+0.3+0(Ext)+0.5(LaTeX)=1.35.
        "<think>\nOkay.\n</think>\n<answer>\n$$ \\boxed{Result} \n $$</answer>\n",  # 24: WITH final NL. $$ \boxed \n $$. Expected: 0.4+0.15+0+0.25(Internal)+0(Ext)+0.25(LaTeX: $$, \boxed, \n$$)=1.05.
        "<think>\nOkay.\n</think>\n<answer>\n$$ \n \\boxed{Result} $$</answer>\n",  # 25: WITH final NL. $$ \n \boxed $$. Expected: 0.4+0.15+0+0.25(Internal)+0(Ext)+0.25(LaTeX: $$, \boxed, $$\n)=1.05.
        "<think>\nOkay.\n</think>\n<answer>\n$$\n\n\\boxed{Result}\n\n$$\n</answer>\n",  # 26: WITH final NL. Perfect LaTeX components found (including NLs), bonus awarded. Expected: 0.4+0.15+0+0.3+0(Ext)+0.5(LaTeX)=1.35.
        "<think>Content</think>\n<answer>Content</answer>",  # 27: No internal NLs, NO final NL. Expected: 0.65.
        "<think>\nContent\n</think>\n<answer>\nContent\n</answer>",  # 28: NO final NL perfect external. Expected: 1.05.
        "<think>\nContent\n</think>\n<answer>\n$$\n\\boxed{R}\n$$\nExtra text after</answer>\n",  # 29: WITH final NL. Perfect LaTeX, text after. Expected: 0.4+0.15+0+0.25+0+0.5=1.3.
        "<think>\nContent\n</think>\n<answer>\nExtra text before. $$<CR>\n\\boxed{R}<CR>\n$$<CR>\n</answer>\n".replace(
            "<CR>", "\n"
        ),  # 30: WITH final NL. Perfect LaTeX, text before. Expected: 0.4+0.15+0+0.3+0+0.5=1.35.
        "<think>\nContent\n</think>\n<answer>\n$$ \\boxed{R} $$ Other text.</answer>\n",  # 31: WITH final NL. Basic LaTeX, text after. Expected: 0.4+0.15+0+0.25+0+0.2=1.0.
        "<think>\nContent\n</think>\n<answer>\nOther text. $$<CR>\n\\boxed{R}<CR>\n$$<CR>\nMore text.<CR>\n</answer>\n".replace(
            "<CR>", "\n"
        ),  # 32: WITH final NL. Perfect LaTeX, text before/after. Expected: 0.4+0.15+0+0.3+0+0.5=1.35.
        "<think>\nOkay.\n</think>\n<answer>\n$$ abc \\boxed{Result} xyz $$\n</answer>\n",  # 33: WITH final NL. $$ \boxed only. Expected: 0.4+0.15+0+0.3+0+0.2=1.05.
        # New example designed to hit max score
        "<think>\nThinking.\n</think>\n<answer>\n$$\n\\boxed{Result}\n$$\n</answer>",  # 34: NO final NL perfect external, perfect LaTeX block. Expected: 0.4+0.15+0+0.3+0.2+0.5=1.55.
    ]

    # Run with debug=True after implementing the final fix to Perfect External Match logic.
    rewards_debug = format_reward(completions_data_list, debug=True)
    print(
        "\nFinal Refined Rewards:",
        [(i, round(r, 4)) for i, r in enumerate(rewards_debug)],
    )

    # --- Calculate Theoretical Max Possible Score ---
    # Assumes a response that matches the PERFECT_PATTERN_NO_FINAL_NL external structure
    # AND contains all components for the perfect LATEX block within the answer content,
    # with non-empty think content and no extraneous content anywhere.
    # An example is "<think>\nThinking\n</think>\n<answer>\n$$\n\\boxed{Answer}\n$$\n</answer>"
    # Such a response would score:
    # Tag Existence (4 * 0.10) = 0.40
    # Tag Order (2 * 0.05 for pairs + 0.05 for blocks) = 0.15
    # Extraneous Content (None) = 0.00
    # Internal Formatting (think open/close, answer open/close, between blocks) = 0.05*2 + 0.05*2 + 0.10 = 0.30
    # Perfect External Match (NO_FINAL_NL pattern AND ends with </answer>) = 0.20
    # Content (Think not empty, Answer has perfect LaTeX block components) = 0.00
    # LaTeX (all components found + bonus) = (0.1+0.1+0.05+0.05) + 0.20 = 0.50
    # No penalties incurred in this ideal case.
    max_score = (
        R_TAG_EXISTS_ONCE * 4  # 0.40
        + R_CORRECT_PAIR_ORDER * 2  # 0.10
        + R_CORRECT_BLOCK_ORDER  # 0.05
        + R_NEWLINE_AFTER_OPEN * 2  # 0.10
        + R_NEWLINE_BEFORE_CLOSE * 2  # 0.10
        + R_NEWLINE_BETWEEN_BLOCKS  # 0.10
        + R_PERFECT_MATCH_EXTERNAL  # 0.20 (Only for NO_FINAL_NL AND ending tag)
        # + P_EMPTY_CONTENT * 0        # Assuming content is not empty
        # No penalties for duplicate tags, extraneous content, or order violations
        + R_LATEX_DOLLARS_PRESENT  # 0.10
        + R_LATEX_BOXED_PRESENT  # 0.10
        + R_LATEX_NEWLINE_BEFORE_BOXED  # 0.05
        + R_LATEX_NEWLINE_AFTER_BOXED  # 0.05
        + R_LATEX_PERFECT_BLOCK  # 0.20 (Bonus if all above LaTeX are found)
    )
    print(f"\nTheoretical Max Score: {max_score:.4f}")  # Should be 1.55
