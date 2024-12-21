import os
import json

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
#
# # 输入和输出路径
# json_dir = ROOT / 'data/cat_dataset/labels'  # JSON 文件目录
# output_dir = ROOT / 'data/cat_dataset/yolo_labels'  # 输出 YOLO 格式目录
# os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
#
# # 类别映射（根据你的 class_with_id.txt 文件内容调整）
# class_mapping = {"cat": 1}  # 例如，"cat" 的 ID 为 1
#
# # 遍历 JSON 文件
# for json_file in os.listdir(json_dir):
#     if json_file.endswith('.json'):
#         # 打开 JSON 文件
#         with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         # 获取图片的宽度和高度
#         img_width = data['imageWidth']
#         img_height = data['imageHeight']
#
#         # 输出 YOLO 标注文件
#         output_txt_path = os.path.join(output_dir, os.path.splitext(json_file)[0] + '.txt')
#         with open(output_txt_path, 'w') as txt_file:
#             for shape in data['shapes']:
#                 # 获取类别和边界框
#                 label = shape['label']
#                 if label not in class_mapping:
#                     print(f"警告：类别 {label} 未在 class_mapping 中定义，跳过...")
#                     continue
#
#                 class_id = class_mapping[label]
#                 points = shape['points']
#                 x_min, y_min = points[0]  # 左上角
#                 x_max, y_max = points[1]  # 右下角
#
#                 # 转换为 YOLO 格式
#                 x_center = (x_min + x_max) / 2 / img_width
#                 y_center = (y_min + y_max) / 2 / img_height
#                 width = (x_max - x_min) / img_width
#                 height = (y_max - y_min) / img_height
#
#                 # 写入到 YOLO 格式文件
#                 txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
#
# print("转换完成！所有标注已保存到:", output_dir)

#上面脚本负责转换,注意默认的文件夹是labels，yolo_labels重命名为labels即可
#转换完成后，还有个问题，文件类别需要从0开始，运行以下脚本，将类别 ID 1 修改为 0：

import os



# 标注文件路径（训练集和验证集的标注文件夹）
# train_labels_path = ROOT / 'data/cat_dataset/labels/train'
val_labels_path = ROOT / 'data/cat_dataset/labels/val'


def fix_labels(label_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                items = line.strip().split()
                if len(items) > 0:
                    items[0] = "0"  # 将类别 ID 修改为 0
                    new_lines.append(" ".join(items) + "\n")

            # 写回修正后的文件
            with open(file_path, "w") as f:
                f.writelines(new_lines)
            print(f"修正完成: {file_path}")


# 修正训练集和验证集的标注文件
# fix_labels(train_labels_path)
fix_labels(val_labels_path)
