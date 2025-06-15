# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import paddle
from paddle.vision import datasets, transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_dataset(is_train, args):
    """
    构建并返回一个数据集对象。
    
    Args:
        is_train (str): 指定数据集的类型 ('train', 'val', or 'test')。
        args: 包含所有命令行参数的命名空间，如data_path。

    Returns:
        paddle.vision.DatasetFolder: 加载好的数据集对象。
    """
    transform = build_transform(is_train, args)
    data_folder_name = str(is_train)
    root = os.path.join(args.data_path, data_folder_name)
    dataset = datasets.DatasetFolder(root, transform=transform)
    print(f"Dataset '{data_folder_name}': Loaded {len(dataset)} samples from {root}")
    return dataset


def build_transform(is_train, args):
    """
    根据是训练阶段还是评估阶段，构建相应的图像变换流水线。
    
    Args:
        is_train (str): 指定变换的类型 ('train', 'val', or 'test')。
        args: 包含所有命令行参数的命名空间，如input_size, aa等。

    Returns:
        paddle.vision.transforms.Compose: 一个组合了多种图像变换操作的对象。
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == "train":
        # 训练阶段：应用复杂的数据增强
        transform_list = []
        transform_list.append(
            transforms.RandomResizedCrop(size=args.input_size, interpolation="bicubic")
        )
        transform_list.append(transforms.RandomHorizontalFlip())
        if args.color_jitter is not None and (not args.aa or str(args.aa).lower() == "none"):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=args.color_jitter,
                    contrast=args.color_jitter,
                    saturation=args.color_jitter,
                )
            )
        if args.aa and str(args.aa).lower() != "none":
            has_rand_augment = hasattr(transforms, "RandAugment")
            has_auto_augment = hasattr(transforms, "AutoAugment")
            if "rand" in args.aa.lower():
                if has_rand_augment:
                    try:
                        parts = args.aa.split("-")
                        num_ops = 2
                        magnitude = 9
                        for part_idx, part_val in enumerate(parts):
                            if ("m" == part_val[0] and part_val[1:].isdigit() and part_idx > 0 and parts[part_idx - 1] == "rand"):
                                magnitude = int(part_val[1:])
                                break
                        print(f"Using RandAugment with num_ops={num_ops}, magnitude={magnitude} (from '{args.aa}')")
                        transform_list.append(transforms.RandAugment(num_ops=num_ops, magnitude=magnitude))
                    except Exception as e:
                        print(f"Warning: Could not parse timm RandAugment string '{args.aa}'. Using default. Error: {e}")
                        transform_list.append(transforms.RandAugment())
                else:
                    print("Warning: paddle.vision.transforms.RandAugment not found. Skipping.")
            elif "autoaug" in args.aa.lower():
                if has_auto_augment:
                    policy_name = "imagenet"
                    if "cifar10" in args.aa.lower():
                        policy_name = "cifar10"
                    elif "svhn" in args.aa.lower():
                        policy_name = "svhn"
                    print(f"Using AutoAugment with policy: {policy_name} (from '{args.aa}')")
                    transform_list.append(transforms.AutoAugment(policy=policy_name))
                else:
                    print("Warning: paddle.vision.transforms.AutoAugment not found. Skipping.")
            else:
                print(f"Warning: Augmentation policy '{args.aa}' not recognized. Skipping.")
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        if args.reprob > 0:
            if hasattr(transforms, "RandomErasing"):
                transform_list.append(transforms.RandomErasing(prob=args.reprob))
            else:
                print("Warning: paddle.vision.transforms.RandomErasing not found. Skipping.")
        transform = transforms.Compose(transform_list)
        return transform

    # 评估阶段：应用标准的预处理
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation="bicubic"))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)