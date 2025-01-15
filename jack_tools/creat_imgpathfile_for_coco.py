

#运行以下脚本生成 train2017.txt
import os

# 数据集根目录
dataset_dir = r"F:/jack_dataset/coco"
# 定义图片和输出文件路径
image_dirs = {
    "train": os.path.join(dataset_dir, "images/train"),  # 训练集图片路径
    "val": os.path.join(dataset_dir, "images/val")       # 验证集图片路径
}
output_files = {
    "train": os.path.join(dataset_dir, "train2017.txt"),  # 输出的训练集路径文件
    "val": os.path.join(dataset_dir, "val2017.txt")       # 输出的验证集路径文件
}

# 遍历每个数据集（train 和 val）
for dataset_type, image_dir in image_dirs.items():
    output_txt = output_files[dataset_type]
    with open(output_txt, "w") as f:
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):  # 检查图片格式
                    # 计算相对路径并写入文件
                    relative_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    f.write(f"{relative_path.replace(os.sep, '/')}\n")  # 替换为统一的路径分隔符
    print(f"{dataset_type} 文件已生成：{output_txt}")
