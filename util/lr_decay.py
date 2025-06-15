# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import json

def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):
    """
    为分层学习率衰减（Layer-wise Learning Rate Decay）创建参数组。
    此方法将模型的参数划分为多个组，每个组根据其在网络中的深度应用不同的学习率缩放因子。
    
    Args:
        model (nn.Layer): 要进行参数分组的模型。
        weight_decay (float): 权重衰减系数。
        no_weight_decay_list (list): 不应用权重衰减的参数名称列表。
        layer_decay (float): 层衰减率，用于计算每层的学习率缩放因子。

    Returns:
        list: 一个包含多个参数组字典的列表，可直接用于优化器。
    """
    param_group_names = {}
    param_groups = {}
    if hasattr(model, "blocks") and model.blocks is not None:
        num_layers = len(model.blocks) + 1
    elif hasattr(model, "num_layers"):
        num_layers = model.num_layers
    elif hasattr(model, "depth"):
        num_layers = model.depth + 1
    else:
        block_indices = set()
        for n_param, _ in model.named_parameters():
            if n_param.startswith("blocks."):
                block_indices.add(int(n_param.split(".")[1]))
        if block_indices:
            num_layers = max(block_indices) + 2
        else:
            num_layers = 12 + 1
    layer_scales = list(
        (layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    )
    for n, p in model.named_parameters():
        if p.stop_gradient:
            continue
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {"params": []}
            param_groups[group_name] = {
                "params": [],
                "weight_decay": this_decay,
                "learning_rate": this_scale,
                "lr_scale": this_scale,
            }
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    根据参数名称，为其分配一个在ViT模型中的层ID（深度）。
    
    Args:
        name (str): 参数的名称 (e.g., 'blocks.0.norm1.weight')。
        num_layers (int): 模型总的逻辑层数。

    Returns:
        int: 该参数所属的层ID。
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    elif name.startswith("norm") and (not name.startswith("blocks.")):
        return num_layers - 1
    elif name.startswith("head"):
        return num_layers
    else:
        return num_layers - 1