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
    550  # The token length where the reward reaches half of MAX_REWARD (1.0).
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
    base_unit = "Step-by-step thinking process. "  # Approx 5 tokens with gpt-4o encoder
    
    # 基础测试样例
    basic_completions = [
        "<answer>No think tag</answer>",  # 0.0 reward
        "<think></think><answer>Empty think</answer>",  # 0.0 reward
        "<think>Short.</think><answer>F</answer>",  # Very short -> Expect near 0
    ]
    
    # 重点关注 300-1000 词元范围，每 50 词元长度打印一次奖励
    focus_completions = []
    for tokens in range(300, 1050, 50):
        # 计算需要多少个 base_unit 来达到目标词元数
        # 假设每个 base_unit 约 5 个词元
        units_needed = tokens // 5
        focus_completions.append(
            f"<think>\n{base_unit * units_needed}\n</think>\n<answer>Token_{tokens}</answer>"
        )
    
    # 添加一些额外的参考点
    extra_completions = [
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
        + base_unit * 300
        + "\n</think>\n<answer>D</answer>",  # ~1500 tokens -> Expect ~1.995 reward
        "<think>\n"
        + base_unit * 400
        + "\n</think>\n<answer>E</answer>",  # ~2000 tokens -> Expect very close to 2.0
    ]
    
    # 合并所有测试样例
    completions_data = focus_completions + basic_completions + extra_completions

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
    
    # 打印结果
    print("\n=== 重点关注区域 (300-1000 词元) ===")
    focus_range = len(focus_completions)
    for i in range(focus_range):
        comp = completions_data[i]
        think_c = extract_tag_content(comp, "think")
        actual_length = get_token_count(think_c) if think_c else 0
        target_length = 300 + (i * 50)
        
        print(
            f"词元长度 {target_length}: 实际长度={actual_length}, 奖励值={rewards[i]:.4f}"
        )
    
    print("\n=== 基础测试样例 ===")
    for i in range(focus_range, focus_range + len(basic_completions)):
        comp = completions_data[i]
        think_c = extract_tag_content(comp, "think")
        actual_length = get_token_count(think_c) if think_c else 0
        
        print(
            f"样例 {i-focus_range+1}: 实际长度={actual_length}, 奖励值={rewards[i]:.4f}"
        )
    
    print("\n=== 额外参考点 ===")
    for i in range(focus_range + len(basic_completions), len(completions_data)):
        comp = completions_data[i]
        think_c = extract_tag_content(comp, "think")
        actual_length = get_token_count(think_c) if think_c else 0
        
        # 估计长度
        idx = i - (focus_range + len(basic_completions))
        multipliers = [10, 40, 150, 300, 400]  # 额外参考点的乘数
        estimated_length = multipliers[idx] * approx_tokens_per_unit
        
        print(
            f"参考点 {idx+1}: 估计长度~{estimated_length}, 实际长度={actual_length}, 奖励值={rewards[i]:.4f}"
        )
    
    # 奖励函数参考值
    print("\n=== 奖励函数参考值 ===")
    print(f"L0={TARGET_CENTER_LENGTH}, k={STEEPNESS_FACTOR}, MAX_REWARD={MAX_REWARD}")
    print("预期奖励值 (近似值):")
    print("- 50 词元: ~0.001")
    print("- 200 词元: ~0.024")
    print("- 750 词元: ~1.000")
    print("- 1000 词元: ~1.762")
    print("- 1500 词元: ~1.995")
    print("- 2000 词元: ~1.999+")
