from math_verify import parse, verify
from .reward_utils import extract_tag_content, completions_to_lst
from typing import List
import time

# --- Constants ---
REWARD_CORRECT_IN_ANSWER = 2.5  # <answer> 标签正确奖励
REWARD_CORRECT_IN_THINK_ONLY = 2.0  # 只在 <think> 标签正确奖励
PENALTY_INCORRECT_ANSWER = -0.5  # 答案错误惩罚
PENALTY_VERIFICATION_ERROR = -0.1  # 验证异常惩罚


def correctness_reward(
    completions,
    answer: List[str],  # 同一个 answer，重复 num_generations 次
    **kwargs,
) -> List[float]:  # 返回值应为 List[float]
    """
    Calculates the reward based on the mathematical correctness of the answer.
    Prioritizes the <answer> tag, falls back to <think> if answer is incorrect/missing.
    Applies parse() to both completion content and ground truth before verify().

    Args:
        completions: List of model response strings or processable format.
        answer: List of ground truth answer strings.

    Returns:
        A list of correctness reward scores.
    """
    start_time = time.time()

    _completions = completions_to_lst(completions)
    _answers = answer  # 使用传入的 answer 列表

    _question_id = kwargs.get("id")
    _question = kwargs.get("question")
    _question_difficulty = kwargs.get("difficulty")

    # 设定分隔线长度
    line_len = 90
    question_title = (
        " Question "
        + (f"{_question_id[0]} " if _question_id else "")
        + (f"（{_question_difficulty[0]}） " if _question_difficulty else "")
    )
    answer_title = " Answer "
    completion_title_tpl = " Completion {} "

    # 打印 Prompt 部分
    print(
        "\n"
        + "=" * ((line_len - len(question_title)) // 2)
        + question_title
        + "=" * ((line_len - len(question_title) + 1) // 2)
    )
    print(_question[0])
    print("=" * line_len)

    # 打印 Answer 部分
    print(
        "\n"
        + "=" * ((line_len - len(answer_title)) // 2)
        + answer_title
        + "=" * ((line_len - len(answer_title) + 1) // 2)
    )
    print(_answers[0])
    print("=" * line_len)

    # 打印每个 Completion
    for i, completion in enumerate(_completions):
        title = completion_title_tpl.format(i + 1)
        print(
            "\n"
            + "=" * ((line_len - len(title)) // 2)
            + title
            + "=" * ((line_len - len(title) + 1) // 2)
        )
        print(completion)
        print("=" * line_len)

    if len(_completions) != len(_answers):
        raise ValueError("Completions list and answers list must have the same length.")

    def _check_answer_correctness(completion: str, single_answer: str) -> float:
        """检查单个 completion 的正确性分数"""
        # 1. 首先解析标准答案 (只执行一次)
        parsed_gold_1 = parse(single_answer)
        parsed_gold_2 = parse(single_answer.split("####")[-1])
        parsed_gold_3 = parse(single_answer.split("####")[-1] + "%")

        completion_answer_content = extract_tag_content(completion, "answer")
        parsed_completion_answer = None
        answer_parse_ok = False

        # 2. 尝试解析和验证 <answer> 标签内容
        if completion_answer_content:
            try:
                parsed_completion_answer = parse(completion_answer_content)
                if (
                    parsed_completion_answer is not None
                ):  # 检查 parse 是否成功返回有效结果
                    answer_parse_ok = True
                    if (
                        verify(parsed_gold_1, parsed_completion_answer)
                        or verify(parsed_gold_2, parsed_completion_answer)
                        or verify(parsed_gold_3, parsed_completion_answer)
                    ):
                        return REWARD_CORRECT_IN_ANSWER  # 验证成功，返回最高奖励
                    else:
                        # 解析成功，但验证失败 (答案错误)
                        return PENALTY_INCORRECT_ANSWER
                # else: parse 失败 (返回 None 或类似值)，继续检查 think
            except Exception as e:
                print(f"Warning: Error during answer content parsing/verification: {e}")
                # 发生异常，可以给错误惩罚，或者保守起见继续检查 think (这里选择给错误惩罚并停止)
                return PENALTY_VERIFICATION_ERROR

        # 3. 如果 <answer> 未提取到内容，或解析失败，或未验证成功，则尝试检查 <think>
        # 注意：如果上面因为答案错误 (PENALTY_INCORRECT_ANSWER) 或异常 (PENALTY_VERIFICATION_ERROR) 返回了，就不会执行到这里

        completion_think_content = extract_tag_content(completion, "think")
        parsed_completion_think = None
        think_parse_ok = False

        if completion_think_content:
            try:
                parsed_completion_think = parse(completion_think_content)
                if parsed_completion_think is not None:  # 检查 parse 是否成功
                    think_parse_ok = True
                    # *** 关键修正：使用 parsed_completion_think 进行验证 ***
                    if (
                        verify(parsed_gold_1, parsed_completion_think)
                        or verify(parsed_gold_2, parsed_completion_think)
                        or verify(parsed_gold_3, parsed_completion_think)
                    ):
                        # 在 <think> 中验证成功
                        return REWARD_CORRECT_IN_THINK_ONLY
                    else:
                        # <think> 中解析成功，但验证失败 (答案错误)
                        # 只有在 <answer> 部分没有给出错误惩罚时，才在这里给出
                        # （当前逻辑下，如果 answer 错误会直接返回，所以这里可以安全返回）
                        return PENALTY_INCORRECT_ANSWER
                # else: parse 失败，继续到最后返回 0.0
            except Exception as e:
                print(f"Warning: Error during think content parsing/verification: {e}")
                return PENALTY_VERIFICATION_ERROR  # 验证异常惩罚

        # 4. 如果 <answer> 和 <think> 都没有成功验证答案
        #    包括：标签不存在、内容为空、解析失败 (返回None)
        #    或者 answer 验证错误、think 验证错误（这些情况已在上面返回）
        #    理论上能到这里的主要是标签/内容缺失或解析失败的情况
        return 0.0

    # 对每个 completion 和对应的 answer 计算分数
    correctness_scores = [
        _check_answer_correctness(_completions[i], _answers[i])
        for i in range(len(_completions))
    ]

    print("\n" + "=" * 90 + "\n")
    print(
        f"Correctness Rewards: {[round(score, 3) for  i, score in enumerate(correctness_scores)]} ({time.time() - start_time:.3f} s)"
    )
    return correctness_scores
