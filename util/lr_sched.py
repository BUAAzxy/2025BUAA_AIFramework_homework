# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import paddle


def adjust_learning_rate(optimizer, cur_step, args):
    """
    根据当前训练步数，调整优化器的学习率。
    实现了带有预热（warmup）的余弦衰减（cosine decay）学习率调度策略。

    Args:
        optimizer (paddle.optimizer.Optimizer): 要调整的优化器。
        cur_step (float): 当前的训练进度，通常是 (epoch + batch_progress)。
        args: 包含所有超参数的命名空间，如 lr, warmup_epochs, epochs。
    """
    # 预热阶段
    if cur_step < args.warmup_epochs:
        base_lr = args.lr * cur_step / float(args.warmup_epochs)
    # 余弦衰减阶段
    else:
        progress = (cur_step - args.warmup_epochs) / float(
            max(1, args.epochs - args.warmup_epochs)
        )
        base_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 将计算出的学习率应用到优化器的每个参数组
    for pg in optimizer._param_groups:
        scale = pg.get("lr_scale", 1.0)  # 获取该组的学习率缩放因子
        pg["learning_rate"] = base_lr * scale

    # (可选) 记录当前的基础学习率
    if hasattr(args, "logger") and args.logger:
        args.logger.add_scalar("train/lr", base_lr, int(cur_step * 1000))
