import cv2
import numpy as np

# 读取图像
import cv2
import numpy as np

# 读取图像
image = cv2.imread('../data/cat_dataset/images/train/IMG_20210728_205231.jpg')

# 将图像从 BGR 转换为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像数据展平为二维数组
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 定义 K-Means 聚类参数
k = 5  # 分成 5 类
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将中心转换为 8 位整数
centers = np.uint8(centers)
labels = labels.flatten()

# 根据聚类结果生成分割图像
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(image.shape)

# 显示结果
cv2.namedWindow("Segmented Image", cv2.WINDOW_NORMAL)
cv2.imshow("Segmented Image", segmented_image)
cv2.resizeWindow("Segmented Image", 800, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()
