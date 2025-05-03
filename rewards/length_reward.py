import math
from typing import List
import tiktoken
import time
from .reward_utils import completions_to_lst, extract_tag_content

# --- Constants ---
# Sigmoid function parameters for length reward aiming for Max 2.0
# Requirements: ~1.75 score at 1000 tokens, ~2.0 score at 1500 tokens.
MAX_REWARD = 2.0  # The maximum possible reward score.
TARGET_CENTER_LENGTH = (
    750  # The token length where the reward reaches half of MAX_REWARD (1.0).
)
# Adjusted L0 significantly lower to meet requirements.
STEEPNESS_FACTOR = (
    0.008  # Controls how quickly the reward increases around TARGET_CENTER_LENGTH.
)
# Increased k for a much steeper curve to reach ~1.75 by 1000 tokens.


# --- Tiktoken Initialization ---
# (Keep the original tiktoken initialization logic)
try:
    # Attempt to load the precise encoder
    encoder = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    try:
        # Fallback to a common encoder if the specific one fails
        encoder = tiktoken.get_encoding("cl100k_base")
        print(
            "Warning: Could not load tiktoken encoder for gpt-4o. Falling back to cl100k_base."
        )
    except Exception as e:
        # Handle case where even the fallback fails
        print(
            f"CRITICAL WARNING: Failed to load any tiktoken encoder (cl100k_base): {e}. Using character length as fallback."
        )
        encoder = None  # Set encoder to None to indicate fallback


def get_token_count(text: str) -> int:
    """
    Calculates the token count for a given text using the initialized tokenizer.
    Falls back to character count if tokenizer failed to load.
    """
    if encoder:
        # Using disallowed_special=() might be important if special tokens affect length significantly
        # but can cause issues if the text *is* meant to contain them. Test appropriately.
        # For simple text content inside <think>, it's likely safe.
        return len(encoder.encode(text, disallowed_special=()))
    else:
        return len(text) // 4


def length_reward(completions, **kwargs) -> List[float]:
    """
    Calculates a reward based on the length of the content within the <think> tag.
    Uses a sigmoid function aiming for rapid reward increase between 200-1000 tokens,
    reaching ~1.75 by 1000 tokens and ~2.0 by 1500 tokens.

    Args:
        completions: List of completion strings or format processable by completions_to_lst.

    Returns:
        A list of length-based reward scores (max MAX_REWARD).
    """
    start_time = time.time()

    _completions = completions_to_lst(completions)

    def _score_single_completion(completion: str) -> float:
        think_content = extract_tag_content(completion, "think")

        # Handle cases where think tag might be present but empty after stripping whitespace
        if not think_content:
            return 0.0

        think_length = get_token_count(think_content)

        if think_length <= 0:
            return 0.0  # Explicitly handle zero length after tokenization/extraction

        # --- Sigmoid Reward Calculation ---
        # Formula: MaxScore / (1 + exp(-k * (length - L0)))
        # L0 = TARGET_CENTER_LENGTH = 750
        # k = STEEPNESS_FACTOR = 0.008
        # MaxScore = MAX_REWARD = 2.0

        exponent = -STEEPNESS_FACTOR * (think_length - TARGET_CENTER_LENGTH)

        # Use try-except for math.exp for robustness against potential extreme values
        try:
            # Calculate the denominator
            denominator = 1 + math.exp(exponent)
            # Check for potential division by zero if denominator is extremely small (highly unlikely here)
            if denominator < 1e-9:  # Avoid division by zero or near-zero
                sigmoid_reward = MAX_REWARD  # If denominator is near zero, means exp() was huge negative -> length >> L0
            else:
                sigmoid_reward = MAX_REWARD / denominator

        except OverflowError:
            # If exp(exponent) overflows, exponent is very large positive (length << L0)
            # Denominator becomes huge, reward approaches 0
            sigmoid_reward = 0.0
        # UnderflowError (exp(exponent) -> 0) means exponent is very large negative (length >> L0)
        # This case is handled correctly by the standard calculation (denominator -> 1)

        # --- Final Score Clamping ---
        # Ensure reward is strictly within [0.0, MAX_REWARD]
        # This also handles any edge cases from the try-except block
        final_reward = max(0.0, min(sigmoid_reward, MAX_REWARD))

        return final_reward

    # Calculate rewards for all completions
    length_rewards = []
    for c in _completions:
        try:
            length_rewards.append(_score_single_completion(c))
        except Exception as e:
            print(f"Error scoring completion: {e}. Appending 0.0 reward.")
            # Optionally log the problematic completion 'c' here
            length_rewards.append(0.0)

    print(
        f"Length Rewards (L0={TARGET_CENTER_LENGTH}, k={STEEPNESS_FACTOR}): "
        f"{[round(score, 4) for score in length_rewards]} ({time.time() - start_time:.3f} s)"
    )
    return length_rewards


if __name__ == "__main__":
    # Example data structure assumed by completions_to_lst
    # Using lengths designed to test the new curve points: ~200, ~750, ~1000, ~1500+
    base_unit = "Step-by-step thinking process. "  # Approx 5 tokens with gpt-4o encoder

    completions_data = [
        "<think>\n"
        + base_unit * 10
        + "\n</think>\n<answer>A</answer>",  # ~50 tokens -> Expect low reward
        "<think>\n"
        + base_unit * 40
        + "\n</think>\n<answer>A</answer>",  # ~200 tokens -> Expect low reward (~0.024)
        "<think>\n"
        + base_unit * 150
        + "\n</think>\n<answer>B</answer>",  # ~750 tokens -> Expect ~1.0 reward
        "<think>\n"
        + base_unit * 200
        + "\n</think>\n<answer>C</answer>",  # ~1000 tokens -> Expect ~1.76 reward
        "<think>\n"
        + base_unit * 300
        + "\n</think>\n<answer>D</answer>",  # ~1500 tokens -> Expect ~1.995 reward
        "<think>\n"
        + base_unit * 400
        + "\n</think>\n<answer>E</answer>",  # ~2000 tokens -> Expect very close to 2.0
        "<answer>No think tag</answer>",  # 0.0 reward
        "<think></think><answer>Empty think</answer>",  # 0.0 reward
        "<think>Short.</think><answer>F</answer>",  # Very short -> Expect near 0
    ]

    # Make sure tiktoken is available or handled
    if encoder is None and not hasattr(get_token_count, "warned_char_fallback"):
        print(
            "CRITICAL WARNING: tiktoken encoder failed to load. Length calculation will use characters, affecting reward scale significantly."
        )

    rewards = length_reward(completions_data)
    print("\n--- Example Results ---")
    print(f"Target Center Length (L0): {TARGET_CENTER_LENGTH}")
    print(f"Steepness Factor (k): {STEEPNESS_FACTOR}")
    print(f"Max Reward: {MAX_REWARD}")
    print("-" * 20)
    # Use a consistent base unit token count for better length estimates in printout
    # Re-calculate based on the actual base_unit used
    approx_tokens_per_unit = 5  # Update this if base_unit changes significantly
    if encoder:
        try:
            approx_tokens_per_unit = get_token_count(base_unit)
        except:  # Catch potential errors during tokenization if base_unit is unusual
            print("Warning: Could not tokenize base_unit accurately.")

    print(f"Approx tokens per base_unit: {approx_tokens_per_unit}")
    print("-" * 20)

    multipliers = [
        10,
        40,
        150,
        200,
        300,
        400,
        0,
        0,
        1,
    ]  # Approximate multipliers for each example
    for i, comp in enumerate(completions_data):
        think_c = extract_tag_content(comp, "think")
        actual_length = get_token_count(think_c) if think_c else 0
        # Use multiplier only for generated examples, not fixed ones
        estimated_length_str = (
            f"~{multipliers[i] * approx_tokens_per_unit}"
            if i < 6
            else f"{actual_length}"
        )
        if i == 8:  # Handle the 'Short.' case specifically
            estimated_length_str = f"{actual_length}"

        print(
            f"Completion {i+1}: Est. Len={estimated_length_str}, Actual Len={actual_length}, Reward={rewards[i]:.4f}"
        )

    # Expected reward profile (approximate, with k=0.008, L0=750):
    # 1: Len ~50, Reward ~0.001
    # 2: Len ~200, Reward ~0.024
    # 3: Len ~750, Reward ~1.000
    # 4: Len ~1000, Reward ~1.762
    # 5: Len ~1500, Reward ~1.995
    # 6: Len ~2000, Reward ~1.999+
    # 7: Len 0, Reward 0.0000
    # 8: Len 0, Reward 0.0000
    # 9: Len 1, Reward ~0.000 (Very low)
