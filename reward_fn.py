import re
from utils import extract_xml_answer, extract_hash_answer
from math_verify import parse, verify

"""
Correctness_reward_func – 奖励精确的标签匹配。

int_reward_func – 鼓励仅回答整数。

soft_format_reward_func – 检查结构但允许轻微的换行不匹配。

strict_format_reward_func – 确保响应结构与提示相匹配，包括换行符。

xmlcount_reward_func – 确保响应中每个 XML 标签恰好有一个。
"""


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    _completions = [completion[0]["content"] for completion in completions]
    _answers = answer
    parsed_completions = [parse(c) for c in _completions]
    parsed_answers = [parse(a) for a in _answers]
    print(prompts)
    print(len(completions))
    return [
        2.0 if resp == gold else 0.0
        for resp, gold in zip(parsed_completions, parsed_answers)
    ]


def format_reward_func(completions, **kwargs) -> list[float]:
    _completions = [completion[0]["content"] for completion in completions]

    def _calc_format_reward(resp) -> float:
        THINK_START = "<think>"
        THINK_END = "</think>"
        ANSWER_START = "<answer>"
        ANSWER_END = "</answer>"

        PERFECT_PATTERN = r"^<think>\n\s*[^\s][\s\S]*?\n</think>\n<answer>\n\s*[^\s][\s\S]*?\n</answer>\n?$"

        reward = 0

        # 检查每个XML标签的出现次数，确保每个标签只出现一次
        for tag in [THINK_START, THINK_END, ANSWER_START, ANSWER_END]:
            count = resp.count(tag)
            if count == 1:
                reward += 0.125  # 每个标签正好出现一次加分
            elif count > 1:
                reward -= 0.25 * (count - 1)  # 每个标签多出现一次扣分

        # 检查标签之间的顺序是否正确
        if THINK_START in resp and THINK_END in resp:
            if resp.find(THINK_START) < resp.find(THINK_END):
                reward += 0.05

        if ANSWER_START in resp and ANSWER_END in resp:
            if resp.find(ANSWER_START) < resp.find(ANSWER_END):
                reward += 0.05

        # 检查整体顺序
        if THINK_END in resp and ANSWER_START in resp:
            if resp.find(THINK_END) < resp.find(ANSWER_START):
                reward += 0.05

        # 惩罚 <think> 前输出内容
        if THINK_START in resp:
            before_think = resp.split(THINK_START, 1)[0]
            # 如果<think>前有非空内容，则扣分
            if before_think.strip():
                # 根据内容长度扣分，内容越多扣分越多
                reward -= 0.1 + len(before_think.strip()) * 0.001

        # 惩罚 </answer> 后仍输出
        if ANSWER_END in resp:
            after_answer = resp.split(ANSWER_END, 1)[1]
            # 如果</answer>后有非空内容，则扣分
            if after_answer.strip():
                # 根据内容长度扣分，内容越多扣分越多
                reward -= 0.1 + len(after_answer.strip()) * 0.001

        if THINK_START in resp and THINK_END in resp:
            think_content = resp.split(THINK_START, 1)[1].split(THINK_END, 1)[0]
            # 检查<think>后是否有换行
            if think_content.startswith("\n"):
                reward += 0.05
            # 检查</think>前是否有换行
            if think_content.rstrip().endswith("\n"):
                reward += 0.05

        # 检查 </think> 和 <answer> 之间是否有换行
        if THINK_END in resp and ANSWER_START in resp:
            between_content = resp.split(THINK_END, 1)[1].split(ANSWER_START, 1)[0]
            # 检查</think>后是否有换行，且<answer>前也有换行
            if between_content.strip() == "" and "\n" in between_content:
                reward += 0.1
            # 如果之间有其他内容，则扣分
            elif between_content.strip() != "":
                reward -= 0.1 + len(between_content.strip()) * 0.001

        if ANSWER_START in resp and ANSWER_END in resp:
            answer_content = resp.split(ANSWER_START, 1)[1].split(ANSWER_END, 1)[0]
            # 检查<answer>后是否有换行
            if answer_content.startswith("\n"):
                reward += 0.05
            # 检查</answer>前是否有换行
            if answer_content.rstrip().endswith("\n"):
                reward += 0.05

        # 检查 <think> 内容是否为空
        if THINK_START in resp and THINK_END in resp:
            think_content = resp.split(THINK_START, 1)[1].split(THINK_END, 1)[0].strip()
            if not think_content:
                reward -= 0.2  # 思考内容为空，扣分

        # 检查 <answer> 内容是否为空
        if ANSWER_START in resp and ANSWER_END in resp:
            answer_content = (
                resp.split(ANSWER_START, 1)[1].split(ANSWER_END, 1)[0].strip()
            )
            if not answer_content:
                reward -= 0.2  # 答案内容为空，扣分

        if re.match(PERFECT_PATTERN, resp.strip()):
            reward += 0.2

        return reward

    return [_calc_format_reward(c) for c in _completions]


# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
