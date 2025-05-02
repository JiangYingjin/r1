"""
You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.
"""

"""
You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
The thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively.
This is how your full response should be like:
<think> In this section, show your detailed reasoning process. 
Besides, If numerical calculations or data manipulation would benefit from programming, you can use Python program within your thinking process like this:
<program>
```python
# Here is the Clear, executable Python code 
# Ensure the code is complete and will run without errors
# You can use multiple lines of code
# Add print statements to display relevant results at the end of your code
```
</program>
After each code block, explicitly state: "The result of executing this Python code is: [code output]"
Continue your reasoning based on these results until you reach a conclusion. 
</think>
<answer> In this section, give your answer based on the thinking steps. Please ensure that your final answer is enclosed within \boxed{}</answer>.
"""

# SYSTEM_PROMPT = """
# The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
# Respond in the following format:
# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

SYSTEM_PROMPT = """
Solve the following math problem. Think step-by-step, consistently verifying, re-evaluating, reconsidering, and checking your reflections as deeply as possible throughout the solving process. The problem will be given in the next prompt. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively.

<think>

1.  **Understand the Problem:** What is the core question being asked? What information is given? Are there any constraints or conditions? What are the key mathematical concepts involved?
2.  **Initial Approach/Strategy:** Based on my understanding, what's a possible way to approach this problem? What formulas, theorems, or techniques might be relevant?
3.  **Step-by-Step Execution:** Execute the chosen strategy step-by-step.
    *   Step 3.1: [Describe the first step of execution]
    *   Step 3.2: [Describe the second step]
    *   ... (Continue for all necessary steps)
4.  **Verification/Checking (Internal):** After each significant step or calculation, does this result make sense in the context of the problem? Are there any obvious errors? Can I quickly check this calculation?
5.  **Re-evaluation/Reconsideration (If Stuck or Doubtful):** If I encounter a roadblock, get an unexpected result, or feel unsure, what went wrong? Is my initial approach flawed? Are there alternative strategies I could try? Let me revisit my understanding of the problem and the relevant concepts.
6.  **Deep Reflection on Reasoning:** Am I making any assumptions that aren't explicitly stated or logically derived? Is my logic sound at each step? Could there be edge cases or special conditions I haven't considered? What are the potential pitfalls in this type of problem? How does this problem relate to other problems I've solved?
7.  **Final Calculation/Derivation:** Complete the final calculations or derivations to arrive at a potential solution.
8.  **Final Verification/Checking (External):** Does the final answer directly address the question asked? Does it make sense in the real-world context (if applicable)? Can I plug the answer back into the original problem to see if it holds true? Are there any alternative ways to check the answer?
9.  **Refinement of Explanation:** How can I clearly articulate the steps taken and the reasoning behind them? Ensure the explanation is logical and easy to follow.

</think>

<answer>

[Present the step-by-step solution based on the thinking process, clearly showing the steps taken.]

The final answer is $\boxed{[Your Final Answer]}$.

</answer>
"""


def extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
