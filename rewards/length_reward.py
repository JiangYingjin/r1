import math
from .reward_utils import extract_tag_content, completions_to_lst
from typing import List
import tiktoken

# --- Constants ---
# 长度相关的参数 (基于字符数，可根据需要调整为 Token 数)
TARGET_THINK_LENGTH = 1500  # 一个理想的思考长度参考值
MAX_REWARD_AT_TARGET = 0.6  # 在理想长度附近期望达到的最大奖励值
PENALTY_START_LENGTH = 2000  # 开始施加惩罚的长度阈值
EXCESS_LENGTH_PENALTY_FACTOR = 0.001  # 每超出1个字符的惩罚系数

# 计算对数函数的缩放因子，使得在 TARGET_THINK_LENGTH 附近达到 MAX_REWARD_AT_TARGET
# 使用自然对数 ln(x+1) 来避免 log(0)
# 我们希望 scale * ln(TARGET_THINK_LENGTH + 1) ≈ MAX_REWARD_AT_TARGET
LOG_SCALE_FACTOR = (
    MAX_REWARD_AT_TARGET / math.log(TARGET_THINK_LENGTH + 1)
    if TARGET_THINK_LENGTH > 0
    else 0
)


# --- Tiktoken Initialization ---
try:
    encoder = tiktoken.encoding_for_model("gpt-4o")
except Exception as e:
    print(
        f"Warning: Could not load tiktoken encoder for gpt-4o: {e}. Falling back to cl100k_base."
    )


def get_token_count(text: str) -> int:
    """
    Calculates the token count for a given text using the GPT-4o tokenizer.
    """
    return len(encoder.encode(text, disallowed_special=()))


def length_reward(completions, **kwargs) -> List[float]:
    """
    Calculates a reward based on the length of the content within the <think> tag.
    Encourages longer thinking processes up to a point, penalizes excessive length.

    Args:
        completions: List of completion strings or format processable by completions_to_lst.

    Returns:
        A list of length-based reward scores.
    """
    _completions = completions_to_lst(completions)

    def _score_single_completion(completion: str) -> float:
        think_content = extract_tag_content(completion, "think")

        if not think_content:
            return 0.0  # No think tag or empty content

        think_length = get_token_count(think_content)

        # 1. Logarithmic reward based on length
        # Using log(length + 1) to handle length 0 gracefully and ensure diminishing returns
        log_reward = LOG_SCALE_FACTOR * math.log(think_length + 1)

        # 2. Penalty for excessive length
        penalty = 0.0
        if think_length > PENALTY_START_LENGTH:
            excess_length = think_length - PENALTY_START_LENGTH
            penalty = EXCESS_LENGTH_PENALTY_FACTOR * excess_length

        # 3. Combine reward and penalty
        # We can cap the positive reward part if desired, e.g., max(log_reward, MAX_REWARD_AT_TARGET)
        # For now, let the log naturally plateau, penalty reduces it further
        final_reward = log_reward - penalty

        # 4. Ensure reward stays within reasonable bounds (optional but recommended)
        # Prevent extremely negative scores from penalty, cap max positive score
        final_reward = max(
            -0.5, min(final_reward, MAX_REWARD_AT_TARGET + 0.1)
        )  # Example bounds

        return final_reward

    return [_score_single_completion(c) for c in _completions]


if __name__ == "__main__":
    # Example data structure assumed by completions_to_lst
    completions_data = [
        "<think>\nShort thought.\n</think>\n<answer>A</answer>",
        "<think>\n"
        + "This is a medium length thought process, exploring options..." * 10
        + "\n</think>\n<answer>B</answer>",
        "<think>\n"
        + "This is a much longer thought process..." * 100
        + "\n</think>\n<answer>C</answer>",  # Approx 3000 chars
        "<think>\n"
        + "Extremely long..." * 200
        + "\n</think>\n<answer>D</answer>",  # Approx 6000 chars
        "<answer>No think tag</answer>",
        "<think></think><answer>Empty think</answer>",
    ]

    rewards = length_reward(completions_data)
    print("Length Rewards:", [round(r, 4) for r in rewards])
