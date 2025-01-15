#验证json文件内容是否符合coco标准
import json
import os


def check_coco_format(annotation_file, image_folder):
    # 加载标注文件
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # 检查关键字段
    assert 'images' in data, "标注文件缺少 'images' 字段"
    assert 'annotations' in data, "标注文件缺少 'annotations' 字段"
    assert 'categories' in data, "标注文件缺少 'categories' 字段"

    # 检查图片是否匹配
    annotated_images = {img['file_name'] for img in data['images']}
    folder_images = {img for img in os.listdir(image_folder) if img.endswith('.jpg')}

    missing_in_annotations = folder_images - annotated_images
    missing_in_folder = annotated_images - folder_images

    print(f"未在标注文件中找到的图片: {missing_in_annotations}")
    print(f"未在文件夹中找到的图片: {missing_in_folder}")


# 替换为实际文件路径
trainval_json = "F:/jack_dataset/coco/Jack_generate_cat/COCO/annotations/trainval.json"
test_json = "F:/jack_dataset/coco/Jack_generate_cat/COCO/annotations/test.json"
train_image_folder = "F:/jack_dataset/coco/Jack_generate_cat/COCO/train2017"
val_image_folder = "F:/jack_dataset/coco/Jack_generate_cat/COCO/val2017"

print("检查 trainval.json:")
check_coco_format(trainval_json, train_image_folder)

print("检查 test.json:")
check_coco_format(test_json, val_image_folder)
