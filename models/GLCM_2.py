
#注意：此文件中的内容并未真正使用，仅仅是测试而已。

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,

)

# -------------------------- 1. 局部 GLCM 提取函数 -------------------------- #
def compute_local_glcm_features(
    x,
    patch_size=3,
    L=8,
    directions=[(0, 1), (1, 0), (1, 1), (1, -1)],
    reduce_mode='average'
):
    """
    在输入特征图 x 上提取局部 GLCM 特征
    x: (B, C, H, W) -- 假设值域在 [0,1] 左右 (或者需先归一化)
    patch_size: 邻域窗口大小 (3,5,...)，stride=1
    L: 灰度级数 (4,8,16,...)，越大越精细，计算量也更高
    directions: 列表，表示要统计的方向(Δr, Δc)
    reduce_mode: 'average' 或 'sum'，用于多方向融合

    返回 feats_4: (B, C*4, H, W)
      - 提取4种特征: contrast, energy, entropy, homogeneity
      - 如果 directions>1 且 reduce_mode='average'，会先把所有方向累加再平均
      - 最终每个通道输出4个特征 => C*4
    """
    B, C, H, W = x.shape
    pad = patch_size // 2

    # 1) Padding，让边缘像素也能取到 patch_size x patch_size 的邻域
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')  # => (B, C, H+2pad, W+2pad)

    # 2) unfold 一次性展开所有 patch
    #    形状变为 (B, C*patch_size*patch_size, H*W)
    unfolded = F.unfold(x_padded, kernel_size=patch_size, stride=1)  # => (B, C * p^2, H*W)
    # 再 reshape => (B*C, p^2, H*W)
    unfolded = unfolded.reshape(B*C, patch_size*patch_size, H*W)  # => (BC, p^2, HW)

    # 3) 量化 [0,1] => [0..L-1], 并 clamp
    unfolded = (unfolded * (L - 1)).long().clamp_(0, L - 1)  # => (BC, p^2, HW)

    # 准备一个 (BC, HW, L, L) 的容器，每个 (H*W) 的patch都会统计一个 (L,L)
    glcm_all_dirs = x.new_zeros((B*C, H*W, L, L))  # float tensor => shape (BC, HW, L, L)

    # 4) 针对多方向统计 GLCM
    for (dr, dc) in directions:
        coords = torch.arange(patch_size * patch_size, device=x.device)
        rr = coords // patch_size
        cc = coords % patch_size

        valid_mask = ((rr + dr) >= 0) & ((rr + dr) < patch_size) & \
                     ((cc + dc) >= 0) & ((cc + dc) < patch_size)
        valid_indices_1 = coords[valid_mask]
        shifted_rr = rr[valid_mask] + dr
        shifted_cc = cc[valid_mask] + dc
        valid_indices_2 = shifted_rr * patch_size + shifted_cc

        v1 = unfolded[:, valid_indices_1, :]  # (BC, #pairs, HW)
        v2 = unfolded[:, valid_indices_2, :]  # (BC, #pairs, HW)

        # 通过 flatten index (v1*L + v2) 来 1D scatter_add
        flat_idx = v1 * L + v2  # 范围在 [0, L^2-1]
        glcm_flat = glcm_all_dirs.view(B*C, H*W, L*L)
        ones_src = torch.ones_like(v1, dtype=glcm_all_dirs.dtype)

        glcm_flat.scatter_add_(
            2,           # 在最后一维 (L^2) 做 scatter
            flat_idx,    # (BC, #pairs, HW)
            ones_src     # (BC, #pairs, HW)
        )

    # 如果是 average 模式，对各方向累加结果做平均
    if reduce_mode == 'average':
        glcm_all_dirs /= len(directions)

    # 5) 归一化 => (BC, HW, L, L)
    glcm_sum = glcm_all_dirs.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)
    glcm_norm = glcm_all_dirs / glcm_sum  # => (BC, HW, L, L)

    # 计算4个纹理特征: contrast, energy, entropy, homogeneity
    idxs = torch.arange(L, device=x.device, dtype=torch.float)
    row_idxs = idxs.view(-1, 1)
    col_idxs = idxs.view(1, -1)
    diff = row_idxs - col_idxs
    diff_sq = diff ** 2
    abs_diff = diff.abs()

    # contrast
    contrast = (diff_sq * glcm_norm).sum(dim=(2,3))
    # energy
    energy   = (glcm_norm * glcm_norm).sum(dim=(2,3))
    # entropy
    entropy  = -(glcm_norm * (glcm_norm + 1e-6).log()).sum(dim=(2,3))
    # homogeneity
    homogene = (glcm_norm / (1.0 + abs_diff)).sum(dim=(2,3))

    # stack => shape (BC, HW, 4)
    feats_4 = torch.stack([contrast, energy, entropy, homogene], dim=-1)  # => (BC, HW, 4)
    # reshape 回 (B, C, H*W, 4) => (B, C*4, H, W)
    feats_4 = feats_4.view(B, C, H*W, 4).permute(0, 1, 3, 2)
    feats_4 = feats_4.reshape(B, C*4, H, W)

    return feats_4




# -------------------------- 2. C3WithGLCM 模块 -------------------------- #

class C3WithGLCM(nn.Module):
    """将 GLCM 特征与 C3 模块相结合的示例"""
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        glcm_channels=4,  # 额外的 GLCM 通道
        patch_size=3,
        L=8,
        directions=[(0,1),(1,0),(1,1),(1,-1)],
        reduce_mode='average'
    ):
        """
        c1: 输入通道
        c2: 输出通道
        n: Bottleneck重复次数
        glcm_channels: 最终会把 GLCM 特征融合成 4 个通道 (contrast, energy, entropy, homogeneity)
                       然后拼接到 x => c1 + 4
        patch_size, L, directions, reduce_mode: 用于 GLCM 计算的参数
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.glcm_channels = glcm_channels
        self.patch_size = patch_size
        self.L = L
        self.directions = directions
        self.reduce_mode = reduce_mode

        # 定义 C3 模块
        self.cv1 = Conv(c1 + glcm_channels, c_, 1, 1)  # 第一条分支
        self.cv2 = Conv(c1 + glcm_channels, c_, 1, 1)  # 第二条分支
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)  # 最终输出

    def compute_glcm(self, x):
        """计算 GLCM 特征，并将 C 维度做 average 得到 (B,4,H,W)"""
        # feats_4c => (B, C*4, H, W)
        feats_4c = compute_local_glcm_features(
            x,
            patch_size=self.patch_size,
            L=self.L,
            directions=self.directions,
            reduce_mode=self.reduce_mode
        )
        # feats_4c => (B, C, 4, H, W)，再对通道 C 做 average => (B,4,H,W)
        B, C, H, W = x.size()
        feats_4c = feats_4c.view(B, C, 4, H, W)
        feats_4 = feats_4c.mean(dim=1)  # => (B,4,H,W)
        return feats_4

    def forward(self, x):
        # 1) 计算 GLCM 特征
        glcm_feats = self.compute_glcm(x)  # => (B,4,H,W)
        # 2) 在通道维拼接
        x_cat = torch.cat([x, glcm_feats], dim=1)  # => (B, C+4, H, W)

        # 3) C3 模块的两条分支
        out1 = self.m(self.cv1(x_cat))  # (B, c_, H, W)
        out2 = self.cv2(x_cat)         # (B, c_, H, W)
        out = torch.cat([out1, out2], dim=1)  # => (B, 2*c_, H, W)
        return self.cv3(out)  # (B, c2, H, W)


# -------------------------- 3. 使用示例 -------------------------- #
if __name__ == "__main__":
    # 假设我们有一个输入张量 (B=1, C=256, H=80, W=80)
    inp = torch.rand(1, 256, 80, 80)

    # 创建一个 C3WithGLCM 模块
    # 输出通道 c2=128, Bottleneck重复次数 n=1, GLCM参数自己调
    model = C3WithGLCM(
        c1=256,
        c2=260,
        n=1,
        glcm_channels=4,  # GLCM 最终特征通道=4
        patch_size=3,
        L=8,
        directions=[(0,1),(1,0)],
        reduce_mode='average'
    )

    # 前向传播
    out = model(inp)
    print(f"Input shape = {inp.shape}, Output shape = {out.shape}")