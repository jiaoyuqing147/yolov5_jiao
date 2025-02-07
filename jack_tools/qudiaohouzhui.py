import os

# 设置文件夹路径
folder_path = r"D:\Jiao\dataset\COCO\Small_traffic_light\visualized_labels_val"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith("_vis.jpg"):  # 只处理 "_vis.jpg" 结尾的文件
        old_path = os.path.join(folder_path, filename)
        new_filename = filename.replace("_vis", "")  # 去掉 "_vis"
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)  # 重命名文件
        print(f"✅ 重命名: {filename} -> {new_filename}")

print("🚀 所有文件重命名完成！")
