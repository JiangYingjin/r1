import re
import math
from .reward_utils import extract_tag_content, completions_to_lst
from typing import List, Set
import time

# --- Constants ---
# 思考关键词列表 (全部小写)
REASONING_KEYWORDS = {
    "but",
    "wait",
    "alternatively",
    "earlier",
    "however",
    "so",
    "if",
    "think",
    "therefore",
    "then",
    "reconsider",
    "rethink",
    "verify",
    "yet",
    "re-evaluate",
    "let's check",
    "let me check",
    "first",
    "firstly",
    "to begin with",
    "second",
    "secondly",
    "next",
    "finally",
    "in conclusion",
    "break this down",
    "break it down",
    "because",
    "since",
    "consequently",
    "as a result",
    "assuming",
    "assumption",
    "given that",
    "examine",
    "hold on",
    "double-check",
    "on the other hand",
    "what if",
    "perhaps",
    "maybe",
    "possibly",
    "another angle",  # 简化多词短语表示
    # 注意: 'let's', 'let me' 本身比较通用，但结合特定开头语境已有处理。
    # 'check' 单独出现也奖励。'assuming'/'assumption' 都包含。
    # 多词短语的处理见下文的 regex 构建。
}
# 强制开头模式
REQUIRED_OPENINGS = [
    "\nOkay, so I need to ",
    "\nOkay, let's ",
    "\nOkay, let me ",
]
OPENING_PENALTY = -0.75  # 如果 <think> 存在但开头错误，施加此惩罚

# 关键词奖励参数
KEYWORD_REWARD_SCALE_FACTOR = 0.08
MAX_KEYWORD_REWARD_CAP = 0.6  # 对此部分奖励设置一个上限

# 构建正则表达式（注意处理多词短语和边界）
# 需要更精细地处理，确保多词短语优先匹配且有边界
# 这是一个简化示例，实际可能需要更复杂的 regex 或分步匹配
keyword_patterns = [
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in REASONING_KEYWORDS
]


def reasoning_reward(completions, **kwargs) -> List[float]:
    """
    Calculates a reward based on the usage of reasoning keywords in <think>
    and penalizes incorrect opening phrases.
    """
    start_time = time.time()

    _completions = completions_to_lst(completions)

    def _score_single_completion(completion: str) -> float:
        think_content_raw = extract_tag_content(
            completion, "think", strip_content=False
        )  # 需要原始content检查开头
        reward = 0.0
        opening_ok = False

        if think_content_raw is not None:  # think 标签存在
            # 1. 检查强制开头
            found_valid_opening = False
            for opening in REQUIRED_OPENINGS:
                if think_content_raw.startswith(opening):
                    found_valid_opening = True
                    opening_ok = True
                    break
            if not found_valid_opening:
                reward += OPENING_PENALTY
                # 开头错了，后续关键词奖励是否还要计算？可以计算，但总分会受惩罚影响
                # 或者选择开头错了直接返回惩罚分？ - 为了提供梯度，还是继续计算

            # 2. 计算关键词奖励 (只有在开头OK或者即使不OK也继续计算时执行)
            think_content_lower = think_content_raw.lower()
            unique_keywords_found = set()

            # 使用编译好的 regex 进行匹配
            for i, pattern in enumerate(keyword_patterns):
                # 获取原始关键词用于添加到set中，避免regex对象
                keyword = list(REASONING_KEYWORDS)[
                    i
                ]  # 注意：依赖set转list顺序，更好的方式是用字典或保持原始列表
                if pattern.search(think_content_lower):
                    unique_keywords_found.add(keyword)  # 添加原始关键词字符串

            if unique_keywords_found:
                keyword_count = len(unique_keywords_found)
                keyword_reward = KEYWORD_REWARD_SCALE_FACTOR * math.sqrt(keyword_count)
                # 对关键词奖励本身也设上限
                keyword_reward = min(keyword_reward, MAX_KEYWORD_REWARD_CAP)
                reward += keyword_reward

        # 确保总奖励不会因为惩罚而过低（如果需要）
        # reward = max(reward, SOME_MINIMUM_VALUE) # 例如 -1.0

        return reward

    reasoning_rewards = [_score_single_completion(c) for c in _completions]
    print(
        f"Reasoning Rewards: {[round(score, 3) for i, score in enumerate(reasoning_rewards)]} ({time.time() - start_time:.3f} s)"
    )
    return reasoning_rewards


# --- 辅助函数 (需要修改或确认 extract_tag_content 支持 strip_content=False) ---
# def extract_tag_content(text: str, tag: str, strip_content: bool = True) -> str | None:
#     start_tag = f"<{tag}>"
#     end_tag = f"</{tag}>"
#     start_index = text.find(start_tag)
#     if start_index == -1:
#         return None
#     start_index += len(start_tag)
#     end_index = text.find(end_tag, start_index)
#     if end_index == -1:
#         return None
#     content = text[start_index:end_index]
#     return content.strip() if strip_content else content
