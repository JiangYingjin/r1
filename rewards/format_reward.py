import re
from typing import List, Dict, Tuple
from rewards.reward_utils import completions_to_lst


def format_reward(completions, **kwargs) -> List[float]:
    """
    Calculates a refined reward score for completions based on adherence
    to the <think>...</think>\n<answer>...</answer> format, with stricter checks.

    Args:
        completions: Input completions, processed by completions_to_lst.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        A list of float reward scores.
    """
    _completions = completions_to_lst(completions)

    # --- Constants ---
    TAG_THINK_START = "<think>"
    TAG_THINK_END = "</think>"
    TAG_ANSWER_START = "<answer>"
    TAG_ANSWER_END = "</answer>"
    REQUIRED_TAGS = [TAG_THINK_START, TAG_THINK_END, TAG_ANSWER_START, TAG_ANSWER_END]

    # Stricter perfect patterns (matched against the raw response)
    PERFECT_PATTERN_NO_FINAL_NL = re.compile(
        r"^<think>\n\s*[^\s][\s\S]*?\n</think>\n<answer>\n\s*[^\s][\s\S]*?\n</answer>$"
    )
    PERFECT_PATTERN_WITH_FINAL_NL = re.compile(
        r"^<think>\n\s*[^\s][\s\S]*?\n</think>\n<answer>\n\s*[^\s][\s\S]*?\n</answer>\n$"
    )

    # --- Rewards ---
    R_TAG_EXISTS_ONCE = 0.125  # Max 0.5 for all 4 tags present exactly once
    R_CORRECT_PAIR_ORDER = 0.05  # <tag> before </tag>
    R_CORRECT_BLOCK_ORDER = 0.05  # </think> before <answer>
    R_NEWLINE_AFTER_OPEN = 0.05
    R_NEWLINE_BEFORE_CLOSE = 0.05
    R_NEWLINE_BETWEEN_BLOCKS = 0.10  # Only newline(s) between </think> and <answer>
    R_PERFECT_MATCH = 0.25  # Increased slightly for stricter matching

    # --- Penalties ---
    P_TAG_DUPLICATED = -0.25  # Per duplicate tag
    P_TAG_MISSING = (
        -0.10
    )  # Flat penalty per missing tag (optional, can rely on lack of R_TAG_EXISTS_ONCE)
    P_CONTENT_OUTSIDE_BASE = -0.10
    P_CONTENT_OUTSIDE_PER_CHAR = -0.005
    P_EMPTY_CONTENT = -0.20  # think or answer block is empty/whitespace only
    P_ORDER_VIOLATION = -0.15  # Specific penalty if tags exist but order is wrong

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
        tag_counts: Dict[str, int],
    ) -> Tuple[float, bool]:
        """Calculates reward/penalty based on tag counts."""
        reward = 0.0
        all_present_once = True
        for tag in REQUIRED_TAGS:
            count = tag_counts.get(tag, 0)
            if count == 1:
                reward += R_TAG_EXISTS_ONCE
            else:
                all_present_once = False
                if count > 1:
                    reward += P_TAG_DUPLICATED * (count - 1)
                # else: # count == 0
                #    Optional: Add flat penalty for missing tags
                #    reward += P_TAG_MISSING
        return reward, all_present_once

    def _check_tag_order(
        tag_positions: Dict[str, int], tag_counts: Dict[str, int]
    ) -> float:
        """Calculates reward/penalty based on tag order."""
        reward = 0.0
        think_start_pos, think_end_pos = (
            tag_positions[TAG_THINK_START],
            tag_positions[TAG_THINK_END],
        )
        answer_start_pos, answer_end_pos = (
            tag_positions[TAG_ANSWER_START],
            tag_positions[TAG_ANSWER_END],
        )

        has_think_pair = (
            tag_counts[TAG_THINK_START] == 1 and tag_counts[TAG_THINK_END] == 1
        )
        has_answer_pair = (
            tag_counts[TAG_ANSWER_START] == 1 and tag_counts[TAG_ANSWER_END] == 1
        )

        # Check internal pair order
        think_order_correct = has_think_pair and think_start_pos < think_end_pos
        answer_order_correct = has_answer_pair and answer_start_pos < answer_end_pos

        if think_order_correct:
            reward += R_CORRECT_PAIR_ORDER
        elif has_think_pair:  # Tags exist but order wrong
            reward += P_ORDER_VIOLATION

        if answer_order_correct:
            reward += R_CORRECT_PAIR_ORDER
        elif has_answer_pair:  # Tags exist but order wrong
            reward += P_ORDER_VIOLATION

        # Check block order (only if both pairs exist and are internally ordered correctly)
        if think_order_correct and answer_order_correct:
            if think_end_pos < answer_start_pos:
                reward += R_CORRECT_BLOCK_ORDER
            else:  # Pairs exist and ordered internally, but block order wrong
                reward += P_ORDER_VIOLATION
        # Note: If pairs don't exist or aren't ordered, block order check is skipped / implicitly penalized

        return reward

    def _penalize_extraneous_content(
        response: str, tag_positions: Dict[str, int], tag_counts: Dict[str, int]
    ) -> float:
        """Calculates penalties for content outside the main tags."""
        penalty = 0.0
        think_start_pos, think_end_pos = (
            tag_positions[TAG_THINK_START],
            tag_positions[TAG_THINK_END],
        )
        answer_start_pos, answer_end_pos = (
            tag_positions[TAG_ANSWER_START],
            tag_positions[TAG_ANSWER_END],
        )

        # Before <think>
        if tag_counts[TAG_THINK_START] > 0 and think_start_pos > 0:
            before_think = response[:think_start_pos].strip()
            if before_think:
                penalty += (
                    P_CONTENT_OUTSIDE_BASE
                    + len(before_think) * P_CONTENT_OUTSIDE_PER_CHAR
                )

        # After </answer>
        if tag_counts[TAG_ANSWER_END] > 0 and answer_end_pos != -1:
            after_answer_start_idx = answer_end_pos + len(TAG_ANSWER_END)
            if after_answer_start_idx < len(response):
                after_answer = response[after_answer_start_idx:].strip()
                if after_answer:
                    penalty += (
                        P_CONTENT_OUTSIDE_BASE
                        + len(after_answer) * P_CONTENT_OUTSIDE_PER_CHAR
                    )

        # Between </think> and <answer>
        # Only penalize if both pairs exist and are correctly ordered internally but have junk between
        think_order_correct = (
            tag_counts[TAG_THINK_START] == 1
            and tag_counts[TAG_THINK_END] == 1
            and think_start_pos < think_end_pos
        )
        answer_order_correct = (
            tag_counts[TAG_ANSWER_START] == 1
            and tag_counts[TAG_ANSWER_END] == 1
            and answer_start_pos < answer_end_pos
        )

        if (
            think_order_correct
            and answer_order_correct
            and think_end_pos < answer_start_pos
        ):
            between_start_idx = think_end_pos + len(TAG_THINK_END)
            between_content = response[between_start_idx:answer_start_pos]
            between_content_stripped = between_content.strip()

            if between_content_stripped != "":  # Found non-whitespace content
                penalty += (
                    P_CONTENT_OUTSIDE_BASE
                    + len(between_content_stripped) * P_CONTENT_OUTSIDE_PER_CHAR
                )
            # Reward for correct newline between is handled separately

        return penalty

    def _check_internal_formatting(
        response: str, tag_positions: Dict[str, int], tag_counts: Dict[str, int]
    ) -> float:
        """Checks newlines around content and empty content penalties."""
        reward = 0.0
        think_start_pos, think_end_pos = (
            tag_positions[TAG_THINK_START],
            tag_positions[TAG_THINK_END],
        )
        answer_start_pos, answer_end_pos = (
            tag_positions[TAG_ANSWER_START],
            tag_positions[TAG_ANSWER_END],
        )

        think_order_correct = (
            tag_counts[TAG_THINK_START] == 1
            and tag_counts[TAG_THINK_END] == 1
            and think_start_pos < think_end_pos
        )
        answer_order_correct = (
            tag_counts[TAG_ANSWER_START] == 1
            and tag_counts[TAG_ANSWER_END] == 1
            and answer_start_pos < answer_end_pos
        )

        # Think block internal format
        if think_order_correct:
            content_start = think_start_pos + len(TAG_THINK_START)
            think_content = response[content_start:think_end_pos]
            if think_content.startswith("\n"):
                reward += R_NEWLINE_AFTER_OPEN
            if think_content.endswith("\n"):
                reward += R_NEWLINE_BEFORE_CLOSE
            if not think_content.strip():
                reward += P_EMPTY_CONTENT

        # Answer block internal format
        if answer_order_correct:
            content_start = answer_start_pos + len(TAG_ANSWER_START)
            answer_content = response[content_start:answer_end_pos]
            if answer_content.startswith("\n"):
                reward += R_NEWLINE_AFTER_OPEN
            if answer_content.endswith("\n"):
                reward += R_NEWLINE_BEFORE_CLOSE
            if not answer_content.strip():
                reward += P_EMPTY_CONTENT

        # Newline(s) between blocks
        if (
            think_order_correct
            and answer_order_correct
            and think_end_pos < answer_start_pos
        ):
            between_start_idx = think_end_pos + len(TAG_THINK_END)
            between_content = response[between_start_idx:answer_start_pos]
            if between_content.strip() == "" and "\n" in between_content:
                reward += R_NEWLINE_BETWEEN_BLOCKS
            # Penalty for non-whitespace content handled in _penalize_extraneous_content

        return reward

    def _check_perfect_match(response: str) -> float:
        """Checks if the response perfectly matches the desired patterns."""
        if PERFECT_PATTERN_NO_FINAL_NL.match(
            response
        ) or PERFECT_PATTERN_WITH_FINAL_NL.match(response):
            return R_PERFECT_MATCH
        return 0.0

    # === Main Calculation Logic ===
    final_rewards = []
    for response in _completions:
        total_reward = 0.0

        # 1. Get tag counts and positions
        tag_counts = _count_tags(response)
        tag_positions = _find_tags(response)  # Find first occurrences

        # 2. Check tag existence and count
        existence_reward, _ = _check_tag_existence_and_count(tag_counts)
        total_reward += existence_reward

        # Only proceed with detailed checks if tags potentially form pairs
        # Note: Order/Content checks internally verify counts are exactly 1 where needed.
        # Basic check here prevents errors if tags are totally absent.
        if (
            tag_counts[TAG_THINK_START] > 0
            and tag_counts[TAG_THINK_END] > 0
            and tag_counts[TAG_ANSWER_START] > 0
            and tag_counts[TAG_ANSWER_END] > 0
        ):

            # 3. Check tag order (pairs and blocks)
            total_reward += _check_tag_order(tag_positions, tag_counts)

            # 4. Penalize extraneous content
            total_reward += _penalize_extraneous_content(
                response, tag_positions, tag_counts
            )

            # 5. Check internal formatting (newlines, emptiness)
            total_reward += _check_internal_formatting(
                response, tag_positions, tag_counts
            )

        else:
            # If essential tags are missing, apply penalties for any existing content?
            # Option: Penalize all content if structure is fundamentally broken.
            # For now, rely on missing tag rewards and potential duplicate penalties.
            pass

        # 6. Check for perfect pattern match (applied to the original string)
        # This is a bonus on top if everything else aligns perfectly.
        total_reward += _check_perfect_match(response)

        # 7. Apply reward floor
        final_rewards.append(max(MIN_REWARD_FLOOR, total_reward))

    return final_rewards


if __name__ == "__main__":
    # --- Example Usage (Using the same examples as before) ---
    completions_data_1 = [
        [
            {
                "role": "assistant",
                "content": "<think>\n  Thinking process here.\n</think>\n<answer>\n  Final answer here.\n</answer>",
            }
        ],  # 0: Near perfect, no final NL
        [
            {
                "role": "assistant",
                "content": "Oops<think>\nThink\n</think>\n<answer>Ans</answer>",
            }
        ],  # 1: Junk before, missing NLs
        [
            {"role": "assistant", "content": "<think>\n </think>\n<answer>\n </answer>"}
        ],  # 2: Empty content, missing final NL
        [
            {"role": "assistant", "content": "<think>T</think><answer>A</answer>"}
        ],  # 3: No NLs
        [
            {
                "role": "assistant",
                "content": "<think>\n T \n</think> \n <answer>\n A \n</answer>\n",
            }
        ],  # 4: Correct spacing between, perfect match likely
        [
            {
                "role": "assistant",
                "content": "<think>\n T \n</think> EXTRA JUNK <answer>\n A \n</answer>\n",
            }
        ],  # 5: Junk between
        [
            {
                "role": "assistant",
                "content": "<think>\n T \n</think>\n<answer>\n A \n</answer>EXTRA AFTER",
            }
        ],  # 6: Junk after
        [
            {
                "role": "assistant",
                "content": "<think>\n T \n</think>\n<answer>\n A \n</answer>\n",
            }
        ],  # 7: Perfect match candidate
        [
            {
                "role": "assistant",
                "content": "<think>\nT\n</think>\n<answer>\nA\n</answer>",
            }
        ],  # 8: Perfect match candidate (no final NL)
        [
            {
                "role": "assistant",
                "content": "<think>\n T \n</think>\n<answer>\n A \n</answer>\n\n",
            }
        ],  # 9: Extra final NL, fails perfect match
    ]

    completions_data_2 = [
        "<think>\nGood\n</think>\n<answer>\nOkay\n</answer>",
        "Bad format",
    ]  # 10, 11

    rewards1 = format_reward(completions_data_1)
    rewards2 = format_reward(completions_data_2)

    # Print rounded results for easier comparison
    print("Refined Rewards 1:", [round(r, 4) for r in rewards1])
    print("Refined Rewards 2:", [round(r, 4) for r in rewards2])

    # # --- Calculate Max Possible Score ---
    # # Assume perfect structure, content, newlines, matching PERFECT_PATTERN_WITH_FINAL_NL
    # max_score = (
    #     R_TAG_EXISTS_ONCE * 4  # 0.5
    #     + R_CORRECT_PAIR_ORDER * 2  # 0.1
    #     + R_CORRECT_BLOCK_ORDER  # 0.05
    #     + R_NEWLINE_AFTER_OPEN * 2  # 0.1
    #     + R_NEWLINE_BEFORE_CLOSE * 2  # 0.1
    #     + R_NEWLINE_BETWEEN_BLOCKS  # 0.1
    #     + R_PERFECT_MATCH  # 0.25
    # )
    # print(f"\nTheoretical Max Score: {max_score:.4f}")  # Should be 1.20
