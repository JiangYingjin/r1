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
        # disallowed_special=() 确保特殊token被正常处理而不是引发错误
        return len(encoder.encode(text, disallowed_special=()))
    except Exception as e:
        print(f"Warning: Tiktoken encoding failed for text snippet: {e}")
        return len(text) // 4  # Fallback heuristic: approx chars/4


# --- Main Self-Contained Reward Function ---
def reasoning_efficiency_reward(
    completions, answer: List[str], **kwargs  # 保持函数签名不变
) -> List[float]:  # 保持返回值类型不变
    """
    Calculates a self-contained, continuous reasoning efficiency reward.

    Internally checks answer correctness (using parse before verify) and
    <think> block token length, then applies a smooth interpolation
    between reward/penalty boundaries.

    Args:
        completions: List of model response strings (or processable by completions_to_lst).
        answer: List of ground truth answer strings. # 注意：参数名是 answer，对应数据集的答案列
        **kwargs: Catches any other potential arguments passed.

    Returns:
        A list of reasoning efficiency reward scores.
    """
    _completions = completions_to_lst(completions)
    _answers = answer  # 使用传入的 answer 列表

    if not (len(_completions) == len(_answers)):
        # 注意：你之前的代码签名中没有 problems，所以这里只比较 completions 和 answers
        raise ValueError(
            "Input lists (completions, answers) must have the same length."
        )

    efficiency_rewards = []

    for idx in range(len(_completions)):
        completion = _completions[idx]
        single_answer = _answers[idx]  # 获取当前对应的标准答案

        # --- Internal Calculation Steps ---

        # 1. Extract Content
        think_content = extract_tag_content(completion, "think")
        answer_content = extract_tag_content(completion, "answer")

        # 2. Calculate Think Length (Tokens)
        think_length_tokens = get_token_count(think_content)

        # --- 3. Determine Correctness (Internal Check with PARSE FIRST) ---
        is_correct = False
        parsed_gold = None  # 用于存储解析后的标准答案

        try:
            # 尝试解析标准答案
            parsed_gold = parse(single_answer)
            if parsed_gold is None:  # 检查 parse 是否成功解析标准答案
                print(f"Warning: Failed to parse ground truth answer: {single_answer}")
                # is_correct 保持 False
        except Exception as e:
            print(f"Warning: Error parsing ground truth answer '{single_answer}': {e}")
            # is_correct 保持 False

        # 只有在标准答案成功解析后，才继续检查模型的输出
        if parsed_gold is not None:
            # 检查 <answer> 标签内容
            if answer_content:
                try:
                    parsed_completion_answer = parse(answer_content)
                    # 检查模型输出的 answer 内容是否成功解析
                    if parsed_completion_answer is not None:
                        # *** 关键修正：使用解析后的对象进行验证 ***
                        if verify(parsed_gold, parsed_completion_answer):
                            is_correct = True  # 标记为正确
                except Exception as e:
                    print(
                        f"Warning: Exception during answer content parsing/verification: {e}"
                    )
                    # 解析或验证异常，视为不正确，is_correct 保持 False

            # 只有在 <answer> 没有验证成功的情况下，才检查 <think> 标签内容
            if not is_correct and think_content:
                try:
                    parsed_completion_think = parse(think_content)
                    # 检查模型输出的 think 内容是否成功解析
                    if parsed_completion_think is not None:
                        # *** 关键修正：使用解析后的对象进行验证 ***
                        if verify(parsed_gold, parsed_completion_think):
                            is_correct = True  # 标记为正确
                except Exception as e:
                    print(
                        f"Warning: Exception during think content parsing/verification: {e}"
                    )
                    # 解析或验证异常，视为不正确，is_correct 保持 False
        # --- End of Correctness Check ---

        # 4. Calculate the Weight using the logistic function (逻辑保持不变)
        exponent = TRANSITION_STEEPNESS * (think_length_tokens - PIVOT_LENGTH)
        weight = 1.0 / (1.0 + math.exp(exponent))

        # 5. Calculate final reward using interpolation (逻辑保持不变)
        reward = 0.0
        if is_correct:
            reward = (
                weight * REWARD_EFFICIENT_CORRECT
                + (1.0 - weight) * REWARD_THOROUGH_CORRECT
            )
        else:
            reward = (
                weight * PENALTY_LAZY_INCORRECT
                + (1.0 - weight) * PENALTY_EFFORTFUL_INCORRECT
            )

        # Optional: Clipping reward (逻辑保持不变)
        # reward = max(PENALTY_LAZY_INCORRECT - 0.1, min(reward, REWARD_EFFICIENT_CORRECT + 0.1))

        efficiency_rewards.append(reward)
        # --- End of Single Completion Calculation ---

    return efficiency_rewards  # 保持返回值不变
