import cv2
import numpy as np

# 读取图像
image = cv2.imread('../data/cat_dataset/images/train/IMG_20210728_205231.jpg')

# 转为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 自适应二值化
binary_image = cv2.adaptiveThreshold(
    gray_image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    91,
    5
)

# 形态学操作清理噪声
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制边界框
output_image = image.copy()
for contour in contours:
    # 获取外接矩形
    x, y, w, h = cv2.boundingRect(contour)
    if cv2.contourArea(contour) > 500:  # 过滤小轮廓
        # 绘制边界框
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 显示结果
cv2.namedWindow("Bounding Boxes", cv2.WINDOW_NORMAL)
cv2.imshow("Bounding Boxes", output_image)
cv2.resizeWindow("Bounding Boxes", 800, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()
