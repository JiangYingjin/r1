import math
import tiktoken
from typing import List, Tuple, Optional
from math_verify import parse, verify
from .reward_utils import completions_to_lst, extract_tag_content

# --- Constants ---
# Reward/Penalty boundaries for efficiency calculation
REWARD_EFFICIENT_CORRECT = 0.7  # Target reward for short thought + correct answer
REWARD_THOROUGH_CORRECT = 0.4  # Target reward for long thought + correct answer
PENALTY_EFFORTFUL_INCORRECT = -0.1  # Target penalty for long thought + incorrect answer
PENALTY_LAZY_INCORRECT = -0.8  # Target penalty for short thought + incorrect answer

# Continuous function parameters
PIVOT_LENGTH = 1000  # Transition center point (token count)
TRANSITION_STEEPNESS = 0.01  # Controls transition steepness (k value)

# --- Tiktoken Initialization ---
try:
    encoder = tiktoken.encoding_for_model("gpt-4o")
except Exception as e:
    print(
        f"Warning: Could not load tiktoken encoder for gpt-4o: {e}. Falling back to cl100k_base."
    )
    encoder = tiktoken.get_encoding("cl100k_base")


def get_token_count(text: Optional[str]) -> int:
    """Calculates token count for given text, handling None input."""
    if not text:
        return 0
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except Exception as e:
        print(f"Warning: Tiktoken encoding failed for text snippet: {e}")
        return len(text) // 4  # Fallback heuristic: approx chars/4


# --- Main Self-Contained Reward Function ---


def calculate_reasoning_efficiency_reward_self_contained(
    completions, answer: List[str], **kwargs
) -> List[float]:
    """
    Calculates a self-contained, continuous reasoning efficiency reward.

    Internally checks answer correctness and <think> block token length,
    then applies a smooth interpolation between reward/penalty boundaries.

    Args:
        completions: List of model response strings (or processable by completions_to_lst).
        problems: List of original problem strings.
        answers: List of ground truth answer strings.

    Returns:
        A list of reasoning efficiency reward scores.
    """
    _completions = completions_to_lst(completions)
    _answers = answer

    if not (len(_completions) == len(_answers)):
        raise ValueError(
            "Input lists (completions, problems, answers) must have the same length."
        )

    efficiency_rewards = []

    for idx in range(len(_completions)):
        completion = _completions[idx]
        answer = _answers[idx]

        # --- Internal Calculation Steps for a Single Completion ---

        # 1. Extract Content
        think_content = extract_tag_content(completion, "think")
        answer_content = extract_tag_content(completion, "answer")

        # 2. Calculate Think Length (Tokens)
        think_length_tokens = get_token_count(think_content)

        # 3. Determine Correctness (Internal Check)
        is_correct = False
        # Check answer tag first
        if answer_content:
            try:
                # Note: Depending on verify/parse, you might need the raw or parsed content.
                # Using raw content here for simplicity, adjust if parse is needed before verify.
                if verify(
                    answer, answer_content
                ):  # Assuming verify(ground_truth, model_output)
                    is_correct = True
            except Exception as e:
                # Log error if needed, but treat as incorrect for reward calculation
                print(f"Warning: Exception during answer content verification: {e}")
                pass

        # If answer wasn't correct, check think tag
        if not is_correct and think_content:
            try:
                if verify(answer, think_content):
                    is_correct = True
            except Exception as e:
                print(f"Warning: Exception during think content verification: {e}")
                pass

        # 4. Calculate the Weight using the logistic function
        exponent = TRANSITION_STEEPNESS * (think_length_tokens - PIVOT_LENGTH)
        weight = 1.0 / (
            1.0 + math.exp(exponent)
        )  # Weight close to 1 for short, close to 0 for long

        # 5. Calculate final reward using interpolation based on correctness and weight
        reward = 0.0
        if is_correct:
            # Interpolate between EFFICIENT (high reward, short) and THOROUGH (lower reward, long)
            reward = (
                weight * REWARD_EFFICIENT_CORRECT
                + (1.0 - weight) * REWARD_THOROUGH_CORRECT
            )
        else:
            # Interpolate between LAZY (high penalty, short) and EFFORTFUL (lower penalty, long)
            reward = (
                weight * PENALTY_LAZY_INCORRECT
                + (1.0 - weight) * PENALTY_EFFORTFUL_INCORRECT
            )

        # Optional: Clipping reward to prevent extreme values outside the defined range
        # reward = max(PENALTY_LAZY_INCORRECT - 0.1, min(reward, REWARD_EFFICIENT_CORRECT + 0.1))

        efficiency_rewards.append(reward)
        # --- End of Single Completion Calculation ---

    return efficiency_rewards


# --- Example Usage ---
# Assume necessary imports and helper functions are available
# completions_data = [...]
# problems_data = [...]
# answers_data = [...]
# efficiency_scores = calculate_reasoning_efficiency_reward_self_contained(
#     completions_data, problems_data, answers_data
# )
# print("Self-Contained Reasoning Efficiency Rewards:", [round(r, 4) for r in efficiency_scores])
