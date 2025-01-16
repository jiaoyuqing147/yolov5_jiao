import os
import xml.etree.ElementTree as ET

# 定义类别字典（根据你的数据集修改类别映射）
class_dict = {"cat": 0}  # 假设数据集只有“cat”类别，类别编号从0开始

def xml_to_txt(xml_folder, txt_folder):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            txt_file = os.path.splitext(xml_file)[0] + ".txt"
            txt_path = os.path.join(txt_folder, txt_file)

            with open(txt_path, "w") as f:
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_id = class_dict.get(class_name)
                    if class_id is not None:
                        bndbox = obj.find("bndbox")
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)

                        # 归一化坐标
                        width = int(root.find("size").find("width").text)
                        height = int(root.find("size").find("height").text)

                        x_center = (xmin + xmax) / 2 / width
                        y_center = (ymin + ymax) / 2 / height
                        obj_width = (xmax - xmin) / width
                        obj_height = (ymax - ymin) / height

                        # 写入YOLO格式
                        f.write(f"{class_id} {x_center} {y_center} {obj_width} {obj_height}\n")

# 使用示例：
xml_folder = r"W:\Jack_datasets\COCO\dataset\Jack_generate_cat\COCO\labels\val2017"  # XML文件路径
txt_folder = r"W:\Jack_datasets\COCO\dataset\Jack_generate_cat\COCO\labels\val2017_yolo"  # YOLO格式的标签存放路径
xml_to_txt(xml_folder, txt_folder)
