#这个文件
import torch
import torch.nn as nn
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class GLCMExtractor(nn.Module):
    def __init__(self, distances=[1], angles=[0], levels=256):
        super(GLCMExtractor, self).__init__()
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def forward(self, x):
        # x: (B, C, H, W) - Batch of grayscale images
        bs, c, h, w = x.shape
        glcm_features = []
        for i in range(bs):
            img = x[i, 0].cpu().numpy().astype(np.uint8)  # Convert to numpy
            glcm = graycomatrix(img, distances=self.distances, angles=self.angles, levels=self.levels, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
            glcm_features.append(features)

        glcm_features = torch.tensor(glcm_features, device=x.device).float()
        return glcm_features
