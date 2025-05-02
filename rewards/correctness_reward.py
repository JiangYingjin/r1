# In rewards/math_correctness_reward.py
from math_verify import parse, verify  # 假设这两个函数可用
from .reward_utils import extract_tag_content, completions_to_lst
from typing import List

# --- Constants ---
REWARD_CORRECT_IN_ANSWER = 2.5  # <answer> 标签正确奖励
REWARD_CORRECT_IN_THINK_ONLY = 2.0  # 只在 <think> 标签正确奖励
PENALTY_INCORRECT_ANSWER = -0.5  # 答案错误惩罚
PENALTY_VERIFICATION_ERROR = -0.1  # 验证异常惩罚


def calculate_math_correctness_reward(
    completions, answer: List[str], **kwargs
) -> float:
    """
    Calculates the reward based on the mathematical correctness of the answer.
    Prioritizes the <answer> tag, falls back to <think> if answer is incorrect/missing.

    Args:
        completion: The full response string from the model.
        answer: The original mathematical answer string.

    Returns:
        The correctness reward score.
    """

    print(completions, answer)

    _completions = completions_to_lst(completions)
    print(_completions)

    _answers = answer  # answer 是数据集参数名

    def _check_answer_correctness(completion: str, answer: str) -> float:
        answer_content = extract_tag_content(completion, "answer")
        parsed_and_verified_answer = False  # 标记 answer 是否已被正确 parse 并 verify

        # 1. 先检查 <answer> 标签
        if answer_content:
            try:
                parsed_answers = parse(answer_content)
                if parsed_answers:  # parse 成功
                    if verify(
                        answer, answer_content
                    ):  # 或 verify(answer, parsed_answers[0])
                        # verify 成功
                        parsed_and_verified_answer = True
                        return REWARD_CORRECT_IN_ANSWER
                    else:
                        # verify 失败，答案错误
                        return PENALTY_INCORRECT_ANSWER
                # parse 失败，继续检查 <think>
            except Exception as e:
                print(f"Warning: Error during answer verification: {e}")
                return PENALTY_VERIFICATION_ERROR  # 验证异常惩罚

        # 2. 如果 <answer> 没有被正确 parse+verify，再检查 <think> 标签
        if not parsed_and_verified_answer:
            think_content = extract_tag_content(completion, "think")
            if think_content:
                try:
                    parsed_thinks = parse(think_content)
                    if parsed_thinks:  # parse 成功
                        if verify(
                            answer, think_content
                        ):  # 或 verify(answer, parsed_thinks[0])
                            # verify 成功
                            return REWARD_CORRECT_IN_THINK_ONLY
                        else:
                            # verify 失败，答案错误
                            return PENALTY_INCORRECT_ANSWER
                    # parse 失败，返回 0.0
                except Exception as e:
                    print(f"Warning: Error during think verification: {e}")
                    return PENALTY_VERIFICATION_ERROR  # 验证异常惩罚

        # 3. 都没命中，返回 0.0
        return 0.0

    return [
        _check_answer_correctness(_completions[i], _answers[i])
        for i in range(len(_completions))
    ]
