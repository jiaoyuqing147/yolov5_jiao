import cv2
import numpy as np

# 读取图像
# image = cv2.imread('../data/cat_dataset/images/train/IMG_20211012_112749.jpg')
image = cv2.imread('../data/cat_dataset/images/train/IMG_20210728_205231.jpg')

# 转为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊，减少噪声
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

# 自适应二值化
binary_image = cv2.adaptiveThreshold(
    blurred_image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    91,  # 调整窗口大小
    3    # 减小 C 值，避免过强二值化
)

# 形态学操作
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 减小闭运算核
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 减小膨胀核
binary_image = cv2.dilate(binary_image, kernel_dilate, iterations=1)  # 减少膨胀次数

# 连通区域分析：过滤小区域
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# 创建一个新图像，只保留较大的连通区域
filtered_image = np.zeros_like(binary_image)
min_area = 500  # 减小最小区域面积阈值
for i in range(1, num_labels):  # 跳过背景
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        filtered_image[labels == i] = 255  # 保留该区域

# 显示中间结果（去噪后的二值图像）
cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
cv2.imshow("Filtered Image", filtered_image)
cv2.resizeWindow("Filtered Image", 800, 600)

# 轮廓检测和优化
contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final_image = np.zeros_like(filtered_image)
cv2.drawContours(final_image, contours, -1, (255), thickness=cv2.FILLED)  # 填充目标区域

# 显示最终结果
cv2.namedWindow("Final Processed Image", cv2.WINDOW_NORMAL)
cv2.imshow("Final Processed Image", final_image)
cv2.resizeWindow("Final Processed Image", 800, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()
