import cv2
import numpy as np

# 读取图像
image = cv2.imread('../data/cat_dataset/images/train/IMG_20210728_205231.jpg')

# 转为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 二值化图像
_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 形态学操作，获取前景和背景
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(binary_image, kernel, iterations=2)
dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 获取未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通区域
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(image, markers)

# 标记边界
image[markers == -1] = [0, 0, 255]

# 显示结果
cv2.namedWindow("Watershed Segmentation", cv2.WINDOW_NORMAL)
cv2.imshow("Watershed Segmentation", image)
cv2.resizeWindow("Watershed Segmentation", 800, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()
