import torch
import torch.nn as nn
from models.yolo import DetectionModel  # 引入原始模型
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from models.GLCM_1 import GLCMExtractor

class GLCMEnhancedDetectionModel(DetectionModel):
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)

        # 定义 GLCM 特征提取模块

        self.glcm_extractor = GLCMExtractor()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        重写 forward 方法，在主干网络中引入 GLCM 特征
        """
        # 原始前向传播
        x = self._forward_once(x, profile, visualize)  # 特征提取

        # 提取 GLCM 特征
        glcm_features = self.glcm_extractor(x[:, :1, :, :])  # 从第一通道提取 GLCM 特征
        glcm_features = glcm_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])

        # 融合 GLCM 特征
        x = torch.cat((x, glcm_features), dim=1)

        return x
