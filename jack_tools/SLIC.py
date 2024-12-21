import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb

# 读取图像
image = cv2.imread('../data/cat_dataset/images/train/IMG_20210728_205231.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用 SLIC 超像素分割
segments = slic(image_rgb, n_segments=200, compactness=10, start_label=1)

# 将 SLIC 结果转为二值化掩码
mask = np.zeros(image.shape[:2], dtype=np.uint8)
for (i, segVal) in enumerate(np.unique(segments)):
    mask[segments == segVal] = 255 if i % 2 == 0 else 0  # 随机选择部分超像素作为初始掩码

# 转换掩码为 GrabCut 格式
# mask == 255 的部分标记为可能的前景 (cv2.GC_PR_FGD)
# mask == 0 的部分标记为可能的背景 (cv2.GC_PR_BGD)
grabcut_mask = np.where(mask == 255, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')

# 初始化背景和前景模型
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 可视化掩码
# 将 GrabCut 格式的掩码值映射为颜色显示
visualized_mask = np.zeros_like(image)

# 前景显示为白色
visualized_mask[grabcut_mask == cv2.GC_FGD] = [255, 255, 255]  # 明确前景
visualized_mask[grabcut_mask == cv2.GC_PR_FGD] = [192, 192, 192]  # 可能前景
# 背景显示为黑色
visualized_mask[grabcut_mask == cv2.GC_BGD] = [0, 0, 0]  # 明确背景
visualized_mask[grabcut_mask == cv2.GC_PR_BGD] = [64, 64, 64]  # 可能背景

# 显示掩码
cv2.namedWindow("GrabCut Mask Visualization", cv2.WINDOW_NORMAL)
cv2.imshow("GrabCut Mask Visualization", visualized_mask)
cv2.resizeWindow("GrabCut Mask Visualization", 800, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 使用 GrabCut 进行分割
rect = None  # 不使用矩形框
cv2.grabCut(image, grabcut_mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)

# 提取前景区域
final_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
result = image * final_mask[:, :, np.newaxis]

# 显示分割结果
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", image)
cv2.resizeWindow("Original Image", 800, 600)

cv2.namedWindow("Segmented Image", cv2.WINDOW_NORMAL)
cv2.imshow("Segmented Image", result)
cv2.resizeWindow("Segmented Image", 800, 600)

cv2.waitKey(0)
cv2.destroyAllWindows()
