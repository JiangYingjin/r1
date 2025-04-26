import matplotlib.pyplot as plt
import numpy as np

# The extracted completion_length values
completion_lengths = [
    268.5,
    421.0,
    370.0,
    389.0,
    214.5,
    422.5,
    342.5,
    222.5,
    232.5,
    344.0,
    245.0,
    270.5,
    347.5,
    351.5,
    371.5,
    219.0,
    229.0,
    301.0,
    357.5,
    372.0,
    290.0,
    185.5,
    256.0,
    230.0,
    230.5,
    211.0,
    255.0,
    261.0,
    241.0,
    378.0,
    176.5,
    396.0,
    216.0,
    274.5,
    246.0,
    286.5,
    241.5,
    226.0,
    266.0,
    323.0,
    246.0,
    217.0,
    328.0,
    379.0,
    375.0,
    198.5,
    121.5,
    352.5,
    277.0,
    232.5,
    302.5,
    358.0,
    207.0,
    226.0,
    216.0,
    307.0,
    305.0,
    267.0,
    249.0,
    136.0,
    299.0,
    281.5,
    290.0,
    356.5,
    315.0,
    246.5,
    291.0,
    339.0,
    343.5,
    224.0,
    167.5,
    360.0,
    232.5,
    347.0,
    308.5,
    192.5,
    351.5,
    304.5,
    161.5,
    286.0,
    135.5,
    211.0,
    239.0,
    373.5,
    311.0,
    187.0,
    270.0,
    249.0,
    250.0,
    204.0,
    139.0,
    354.0,
    255.5,
    230.5,
    219.5,
    246.0,
    304.0,
    246.5,
    278.0,
    220.5,
]

# 创建步骤列表 (1 到 100)
steps = list(range(1, len(completion_lengths) + 1))

# 创建图形
plt.figure(figsize=(15, 6))  # 设置图形大小
plt.plot(
    steps, completion_lengths, marker="o", linestyle="-", markersize=4
)  # 绘制折线图，添加点标记

# 添加标题和标签
plt.title("Completion Length Changes During Training")
plt.xlabel("Training Steps")
plt.ylabel("Completion Length (Number of Tokens)")

# 添加网格线
plt.grid(True)

# 显示图形
plt.show()
