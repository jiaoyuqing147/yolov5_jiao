import cv2
import numpy as np

# 读取图像
image = cv2.imread('../data/cat_dataset/images/train/IMG_20211012_112749.jpg')

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化图像
ret, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 获取连通区域信息
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# 为了避免背景区域被标记，我们从 1 开始标记（0 是背景）
output_image = image.copy()

# 为每个连通区域分配不同的颜色
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

# 遍历所有区域，标记区域
for i in range(1, num_labels):  # 从 1 开始，跳过背景
    # 获取该区域的统计信息
    x, y, w, h, area = stats[i]
    color = colors[i]

    # 在区域周围绘制一个矩形框
    cv2.rectangle(output_image, (x, y), (x + w, y + h), color.tolist(), 2)

    # 可选：在区域中心绘制一个圆
    cx, cy = int(centroids[i][0]), int(centroids[i][1])
    cv2.circle(output_image, (cx, cy), 5, color.tolist(), -1)  # 绘制圆点，大小为 5

# 显示结果
# 保存标记后的图像到本地
cv2.imwrite('labeled_image.jpg', output_image)

# 显示保存后的图像
cv2.imshow("Labeled Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
