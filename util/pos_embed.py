# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
import paddle
import paddle.nn as nn

# --------------------------------------------------------
# 2D Sine-Cosine Position Embedding
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    为2D网格生成正弦-余弦（sincos）位置编码。

    Args:
        embed_dim (int): 位置编码的维度。
        grid_size (int): 网格的边长 (e.g., 14 for 224x224 image with 16x16 patch)。
        cls_token (bool): 是否为CLS token在开头添加一个额外的编码。

    Returns:
        numpy.ndarray: 形状为 (grid_size*grid_size, embed_dim) 或 (1+grid_size*grid_size, embed_dim) 的位置编码。
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从给定的网格坐标生成sincos位置编码。

    Args:
        embed_dim (int): 编码维度。
        grid (numpy.ndarray): 网格坐标。

    Returns:
        numpy.ndarray: 位置编码。
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从1D坐标生成sincos位置编码。

    Args:
        embed_dim (int): 编码维度。
        pos (numpy.ndarray): 1D坐标。

    Returns:
        numpy.ndarray: 1D位置编码。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum("i,j->ij", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# --------------------------------------------------------
# Interpolate Position Embeddings for Fined-tuning
# --------------------------------------------------------


def interpolate_pos_embed(model, checkpoint_model):
    """
    对位置编码进行插值，以适应不同的输入图像尺寸。
    当微调时的输入尺寸与预训练时的尺寸不同时，此函数至关重要。

    Args:
        model (nn.Layer): 当前的模型实例。
        checkpoint_model (dict): 从检查点文件中加载的状态字典。
    """
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # 获取原始的2D位置编码 (去掉CLS token)
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # 获取新的2D位置编码目标尺寸
        new_size = int(num_patches**0.5)

        # 如果尺寸不变，则无需插值
        if orig_size == new_size:
            print("Position interpolate pass")
            return

        print(
            f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}"
        )
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        # 去掉CLS token，只对patch的位置编码进行操作
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        # 将1D的位置编码序列重塑为2D网格形状
        pos_tokens = pos_tokens.reshape(
            [-1, orig_size, orig_size, embedding_size]
        ).transpose([0, 3, 1, 2])

        # 使用双三次插值法（bicubic）调整网格大小
        pos_tokens = nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )

        # 将插值后的2D网格展平回1D序列
        pos_tokens = pos_tokens.transpose([0, 2, 3, 1]).flatten(1, 2)

        # 将CLS token等额外token与插值后的patch编码拼接回来
        new_pos_embed = paddle.concat([extra_tokens, pos_tokens], axis=1)

        # 更新状态字典中的位置编码
        checkpoint_model["pos_embed"] = new_pos_embed
