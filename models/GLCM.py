import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ 工具函数: 计算多方向局部GLCM，并提取4种纹理特征 ------------------ #
def compute_local_glcm_features(
    x,
    patch_size=3,
    L=16,
    directions=[(0,1), (1,0), (1,1), (1,-1)],
    reduce_mode='average'  # 'average' 或 'sum' 或 'concat'
):
    """
    x: (B, C, H, W), float in [0,1] (假设)
    patch_size: GLCM邻域大小(odd number), e.g. 3,5,...
    L: 量化灰度级数
    directions: 列表，表示想统计的像素偏移(direction)，如[(0,1), (1,0), (1,1), (1,-1)]
    reduce_mode: 多方向融合方式
        - 'average': 不同方向的 GLCM 相加平均后再提特征
        - 'sum': 同上但不做平均
        - 'concat': 在特征维度上拼接(可能得到更多channel)，这里为了简洁不做完整示例

    return: glcm_feats => (B, 4, H, W)
      - 这里提取 contrast, energy, entropy, homogeneity 4个特征
      - 若 directions>1 且 reduce_mode='concat', 通道数会变为 4 * num_directions
    """

    B, C, H, W = x.shape
    pad = patch_size // 2

    # 1) 对输入做 padding, 保证边缘像素也能取到 patch_size x patch_size 邻域
    #    例如 'replicate' 或 'reflect'
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')  # (B, C, H+2pad, W+2pad)

    # 2) 用 unfold 一次性提取所有patch: 卷积视角 => kernel_size=patch_size, stride=1
    #    out形状: (B, C*patch_size^2, out_H*out_W), 其中 out_H = H, out_W = W (stride=1)
    unfolded = F.unfold(x_padded, kernel_size=patch_size, stride=1)  # (B, C*ks*ks, H*W)
    # 我们 reshape 为 (B*C, patch_size^2, H*W)
    unfolded = unfolded.reshape(B*C, patch_size*patch_size, H*W)  # => (BC, p^2, HW)

    # 3) 量化到 [0, L-1], 并 clamp
    unfolded = (unfolded * (L - 1)).long().clamp_(0, L - 1)  # (BC, p^2, HW)

    # 准备一个容器，用来存放多方向 GLCM 累加或拼接
    # shape: (BC, HW, L, L) -- 每个 patch 独立一个 (L,L) 矩阵, 共 BC * HW 个
    # 但如果 direction>1, 我们要么做多条目累加, 要么在新维度上存
    glcm_all_dirs = x.new_zeros((B*C, H*W, L, L))  # float tensor

    # ---- 4) 多方向统计: 这里演示对 directions 做累加, 然后可做平均 (reduce_mode='average') ----
    for (dr, dc) in directions:
        # 计算在 patch_size x patch_size 展平后, 相邻像素的索引偏移
        # patch layout: 行优先 => index = r*patch_size + c
        # dr,dc in [-patch_size+1, ..., patch_size-1]

        # 先得到 0..(p^2-1) 的网格坐标
        coords = torch.arange(patch_size*patch_size, device=x.device)
        rr = coords // patch_size  # 行坐标
        cc = coords % patch_size   # 列坐标

        # 目标像素 = (rr+dr, cc+dc)
        # 需要保证不越界(0 <= rr+dr < patch_size, 0 <= cc+dc < patch_size)
        valid_mask = ((rr + dr) >= 0) & ((rr + dr) < patch_size) & \
                     ((cc + dc) >= 0) & ((cc + dc) < patch_size)
        valid_indices_1 = coords[valid_mask]  # 原像素index
        shifted_rr = rr[valid_mask] + dr
        shifted_cc = cc[valid_mask] + dc
        valid_indices_2 = shifted_rr * patch_size + shifted_cc  # 相邻像素index

        # 取出对应的灰度值对 (v1, v2) => shape: (BC, len(valid_indices), HW)
        v1 = unfolded[:, valid_indices_1, :]  # (BC, #pairs, HW)
        v2 = unfolded[:, valid_indices_2, :]  # (BC, #pairs, HW)

        # 现在要把 (v1, v2) 统计到 (L,L) 直方图: glcm_all_dirs[ bc, hw, v1, v2 ] += 1
        # 由于 v1,v2 都是 [0..L-1] 的整数，可用 scatter_add:
        #   scatter_add_(dim, index, src) => glcm[..., index] += src
        # 但是我们是2D binning => 需要在最后两个维度(L,L)都用 index. 下面演示一种方法:
        bc_range = torch.arange(B*C, device=x.device)[:, None, None]  # (BC,1,1)
        hw_range = torch.arange(H*W, device=x.device)[None, None, :]  # (1,1, HW)

        # 构造一个ones张量, shape跟 (v1) 一样, 代表要加1
        ones_src = torch.ones_like(v1, dtype=glcm_all_dirs.dtype)

        # glcm_all_dirs[bc, hw, v1, v2] += 1
        # => 需要先 expand bc_range, hw_range 的形状, 再 scatter_add_
        # v1,v2 也是 (BC, #pairs, HW), 与 bc_range, hw_range 对应
        glcm_all_dirs.scatter_add_(
            2,  # 在 dim=2 上做 scatter => 先选 v1 作为行 (或列)索引
            v1.unsqueeze(-1).expand(-1, -1, -1, 1),  # (BC, #pairs, HW, 1)
            torch.zeros_like(ones_src).unsqueeze(-1)  # 先暂存, 再二次scatter
        )
        # 上面是一种思路，但我们要同时用 v1,v2 做2D scatter。PyTorch 并没有原生 2D scatter_add。
        # 可以通过 trick: flatten (v1,v2) => single index = v1*L + v2 => in [0, L^2-1],
        # 再 scatter_add_ 到 dim=2, 这样 glcm[:,:,(v1*L+v2)] += 1
        # 然后 reshape => (L,L). 下面直接示例这种做法:

        flat_idx = (v1 * L + v2)  # (BC, #pairs, HW), in [0, L^2-1]
        # glcm_all_dirs shape = (BC, HW, L, L) => flatten => (BC, HW, L^2)
        glcm_flat = glcm_all_dirs.view(B*C, H*W, L*L)  # flatten last 2 dims
        # scatter_add in dim=2
        glcm_flat.scatter_add_(
            2,
            flat_idx,  # (BC, #pairs, HW)
            ones_src,   # (BC, #pairs, HW)
        )
        # glcm_all_dirs 已更新

    # directions循环结束 => glcm_all_dirs (BC, HW, L, L) 已包含**所有方向**的累加
    if reduce_mode == 'average':
        glcm_all_dirs /= len(directions)
    elif reduce_mode == 'sum':
        pass
    elif reduce_mode == 'concat':
        # 这里为了简洁没展示；一般你需要另一个 glcm_all_dirs_list 来收集每个方向再concat
        # 这里只是示意
        raise NotImplementedError("demo只写了average/sum，如要concat可自行实现")

    # glcm_all_dirs 形状: (BC, HW, L, L).
    # 现在要对每个patch (BC,HW) 做归一化 => sum over (L,L), compute 4 features => (BC, HW, 4)
    glcm_sum = glcm_all_dirs.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)
    glcm_norm = glcm_all_dirs / glcm_sum  # (BC, HW, L, L)

    # 计算4个纹理特征: contrast, energy, entropy, homogeneity
    # 索引向量 [0..L-1]
    idxs = torch.arange(L, device=x.device).float()
    row_idxs = idxs.view(-1,1)  # (L,1)
    col_idxs = idxs.view(1,-1)  # (1,L)
    diff = row_idxs - col_idxs
    diff_sq = diff ** 2
    abs_diff = diff.abs()

    # (BC, HW, L, L) * broadcasting -> sum in (2,3)
    contrast = (diff_sq * glcm_norm).sum(dim=(2,3))       # (BC, HW)
    energy   = (glcm_norm * glcm_norm).sum(dim=(2,3))     # (BC, HW)
    entropy  = -(glcm_norm * (glcm_norm+1e-6).log()).sum(dim=(2,3))  # (BC, HW)
    homogene = (glcm_norm / (1.0 + abs_diff)).sum(dim=(2,3))  # (BC, HW)

    # 拼成 (BC, HW, 4)
    feats_4 = torch.stack([contrast, energy, entropy, homogene], dim=-1)  # (BC, HW, 4)

    # reshape 回 (B, C, H*W, 4) => (B, C, 4, H, W) => or 直接合并C维?
    feats_4 = feats_4.view(B, C, H*W, 4).permute(0, 1, 3, 2)  # (B, C, 4, HW)
    feats_4 = feats_4.view(B, C*4, H, W)  # 这里演示保留C维: => (B, 4C, H, W)

    # 如果你只想对通道做平均(不区分通道)，则再 reduce dim=1
    # feats_4 = feats_4.view(B, C, 4, H, W).mean(dim=1)  # => (B, 4, H, W)

    return feats_4


# ------------------ 示例: 将上面函数融合到一个自定义 C3WithGLCM 模块 ------------------ #
class C3WithGLCM(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        glcm_channels=4,   # 期望输出多少 GLCM特征通道
        patch_size=3,
        L=16,
        directions=[(0,1), (1,0), (1,1), (1,-1)],
        reduce_mode='average'
    ):
        """
        glcm_channels: 如果最终只想要 4 个特征通道 (contrast, energy, entropy, homogeneity),
                       且对通道C做平均 => glcm_channels=4
                       如果保留通道并输出(4*C), 则glcm_channels应设为 4*C => 由自己掌控
        """
        super().__init__()
        self.glcm_channels = glcm_channels
        self.patch_size = patch_size
        self.L = L
        self.directions = directions
        self.reduce_mode = reduce_mode

        # 原始 C3 结构
        c_ = int(c2 * e)  # Hidden channels
        # 这里假设你会把 glcm_feats 直接拼回原特征 => (B, c1 + glcm_channels, H, W) 送给后续
        self.cv1 = Conv(c1 + glcm_channels, c_, 1, 1)
        self.cv2 = Conv(c1 + glcm_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g) for _ in range(n)))

    def compute_glcm(self, x):
        """
        使用上面定义的 unfold + 多方向GLCM 函数
        这里可以根据 self.reduce_mode 来决定是否对通道做平均
        """
        # feats_4c = shape (B, 4*C, H, W) or (B, 4, H, W) 具体看内部实现
        feats_4c = compute_local_glcm_features(
            x,
            patch_size=self.patch_size,
            L=self.L,
            directions=self.directions,
            reduce_mode='average'  # 先算出(BC, HW, L, L),再融合多方向
        )

        # 如果最后只想要 (B, 4, H, W), 对 C 做平均:
        # feats_4c => (B, 4*C, H, W)
        # reduce channel
        feats_4 = feats_4c.view(x.size(0), x.size(1), 4, x.size(2), x.size(3))
        feats_4 = feats_4.mean(dim=1)  # => (B, 4, H, W)

        return feats_4

    def forward(self, x):
        # 1) 计算 GLCM 特征
        glcm_feats = self.compute_glcm(x)  # => (B, 4, H, W)

        # 2) 拼接 => (B, C+4, H, W)
        x_cat = torch.cat([x, glcm_feats], dim=1)

        # 3) 原始 C3流程
        out = torch.cat([self.m(self.cv1(x_cat)), self.cv2(x_cat)], dim=1)
        out = self.cv3(out)
        return out


# ------------------ 简单测试 ------------------ #
if __name__ == "__main__":
    # 假设输入 (B=2, C=8, H=16, W=16)
    inp = torch.rand(2, 8, 16, 16).cuda()  # 放到GPU看加速效果
    model = C3WithGLCM(
        c1=8, c2=16,
        glcm_channels=4,
        patch_size=3,
        L=16,
        directions=[(0,1), (1,0), (1,1), (1,-1)],
        reduce_mode='average'
    ).cuda()
    out = model(inp)
    print("inp.shape =", inp.shape, "=> out.shape =", out.shape)
    # 期望 out: (B, c2, H, W) => (2, 16, 16, 16)
