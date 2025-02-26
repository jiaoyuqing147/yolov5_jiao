import yaml
from utils.dataloaders import LoadImagesAndLabels
from utils.autoanchor import kmean_anchors

# 1. 读取你的 data.yaml
yaml_path = 'data/cocoTrafficLightAuto.yaml'
with open(yaml_path, 'r', encoding='utf-8') as f:
    data_dict = yaml.safe_load(f)

# 2. 创建 Dataset 对象，并且禁用缓存（cache=False）
dataset = LoadImagesAndLabels(
    path=data_dict["train"],  # data.yaml 里 train 对应的路径
    augment=True,
    rect=True,
    cache_images=False  # 显式禁用缓存
)

# 3. 调用 kmean_anchors，直接传入上面创建的 dataset 对象
anchors = kmean_anchors(
    dataset=dataset,   # 注意这里直接传的是对象，而不是字符串
    n=9,
    img_size=640,
    thr=4.0,
    gen=1000,
    verbose=True
)

print("生成的 anchors:\n", anchors)
