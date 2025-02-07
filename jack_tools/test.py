import kagglehub

# 下载最新版本的数据集
path = kagglehub.dataset_download("wjybuqi/traffic-light-detection-dataset")

print("数据集文件路径:", path)
