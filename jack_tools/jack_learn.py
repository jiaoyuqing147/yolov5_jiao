import matplotlib.pyplot as plt
import numpy as np

# 定义输入特征图 X (2x2x3)
X = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
channels = ['Channel 1', 'Channel 2', 'Channel 3']

for i, ax in enumerate(axes):
    ax.matshow(X[:, :, i], cmap='coolwarm')
    ax.set_title(channels[i])
    for (j, k), val in np.ndenumerate(X[:, :, i]):
        ax.text(k, j, f'{val}', ha='center', va='center', color='black')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Input Feature Map (2x2x3)", fontsize=14)
plt.tight_layout()
plt.show()
