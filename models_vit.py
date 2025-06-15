# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False):
    """
    生成正弦/余弦位置编码表。
    
    Args:
        n_position (int): 序列的最大长度。
        d_hid (int): 编码的维度。
        cls_token (bool): 是否为CLS token额外增加一个位置。

    Returns:
        paddle.Tensor: 生成的位置编码张量，形状为 (1, n_position, d_hid)。
    """
    if cls_token:
        n_position = n_position + 1

    def get_position_angle_vec(position):
        return [
            position / paddle.pow(paddle.to_tensor(10000.0), 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = paddle.to_tensor(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)],
        dtype=paddle.get_default_dtype(),
    )
    sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])
    return sinusoid_table.unsqueeze(0)


class DropPath(nn.Layer):
    """
    实现DropPath（随机深度）正则化。
    在训练过程中，以一定的概率随机“丢弃”整个残差连接，强制模型学习冗余路径。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def _drop_path_impl(self, x, drop_prob: float = 0.0, training: bool = False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
        random_tensor = paddle.to_tensor(keep_prob, dtype=x.dtype) + paddle.rand(
            shape, dtype=x.dtype
        )
        random_tensor = paddle.floor(random_tensor)
        output = x / keep_prob * random_tensor
        return output

    def forward(self, x):
        """执行DropPath的前向传播。"""
        return self._drop_path_impl(x, self.drop_prob, self.training)


class PatchEmbed(nn.Layer):
    """
    图像到块嵌入（Image to Patch Embedding）层。
    将输入的2D图像通过一个卷积层切分成一系列展平的1D块嵌入向量。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]
        )
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """执行PatchEmbed的前向传播。"""
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = paddle.flatten(x, start_axis=2)
        x = paddle.transpose(x, perm=[0, 2, 1])
        return x


class Attention(nn.Layer):
    """
    标准的多头自注意力（Multi-Head Self-Attention）模块。
    Transformer架构的核心，用于捕捉序列内不同位置之间的依赖关系。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """执行多头自注意力的前向传播。"""
        B, N, C = x.shape
        qkv_out = self.qkv(x)
        qkv = qkv_out.reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose(
            [2, 0, 3, 1, 4]
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Layer):
    """
    多层感知器（Multi-Layer Perceptron）模块。
    在Transformer块中用作前馈神经网络（FFN）。
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """执行MLP的前向传播。"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Layer):
    """
    单个Transformer编码器块。
    由一个多头自注意力模块和一个MLP模块组成，每个模块前后都有残差连接和层归一化。
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        """执行Transformer块的前向传播。"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PaddleVisionTransformerBase(nn.Layer):
    """
    视觉变换器（Vision Transformer）的基础实现。
    这个基类封装了ViT的核心组件，如块嵌入、位置嵌入和Transformer块堆叠。
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_cls_token=True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes_for_base_head = num_classes
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        _norm_layer = (
            norm_layer
            if norm_layer is not None
            else partial(nn.LayerNorm, epsilon=1e-6)
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        if self.use_cls_token:
            self.cls_token = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype=paddle.get_default_dtype(),
                default_initializer=nn.initializer.Constant(value=0.0),
            )
        else:
            self.cls_token = None
        num_pos_embed_tokens = num_patches + (1 if self.use_cls_token else 0)
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_pos_embed_tokens, embed_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.TruncatedNormal(std=0.02),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=_norm_layer, act_layer=nn.GELU,
                )
                for i in range(depth)
            ]
        )
        self.norm = _norm_layer(embed_dim)
        if self.num_classes_for_base_head > 0:
            self.head = nn.Linear(embed_dim, self.num_classes_for_base_head)
        else:
            self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化线性层和归一化层权重的函数。"""
        if isinstance(m, nn.Linear):
            nn.initializer.TruncatedNormal(std=0.02)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(value=0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.initializer.Constant(value=0)(m.bias)
            if m.weight is not None:
                nn.initializer.Constant(value=1)(m.weight)

    def _get_pos_embed_runtime(self, pos_embed_param, x_tokens_with_cls):
        """
        在运行时处理位置编码，以应对可能的输入尺寸变化。
        如果输入token数与位置编码的token数不匹配，进行简单的截断或填充。
        理想情况下，位置编码的插值应在加载模型权重之前完成。
        """
        if x_tokens_with_cls.shape[1] != pos_embed_param.shape[1]:
            print(
                f"Warning: Positional embedding shape mismatch. Input tokens: {x_tokens_with_cls.shape[1]}, pos_embed tokens: {pos_embed_param.shape[1]}. "
                "This can lead to errors or unexpected behavior."
            )
            if x_tokens_with_cls.shape[1] < pos_embed_param.shape[1]:
                return pos_embed_param[:, : x_tokens_with_cls.shape[1], :]
            else:
                diff = x_tokens_with_cls.shape[1] - pos_embed_param.shape[1]
                last_embed = pos_embed_param[:, -1:, :].expand([-1, diff, -1])
                return paddle.concat([pos_embed_param, last_embed], axis=1)
        return pos_embed_param

    def forward_features(self, x):
        """
        执行特征提取部分的前向传播（不包括最后的分类头）。
        
        Args:
            x (paddle.Tensor): 输入图像张量。

        Returns:
            paddle.Tensor: 经过Transformer块处理后的特征张量。
        """
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat([cls_tokens, x], axis=1)
        x = x + self._get_pos_embed_runtime(self.pos_embed, x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        """
        执行完整ViT模型的前向传播，包括分类头。
        """
        x_features = self.forward_features(x)
        if self.cls_token is not None:
            outcome = x_features[:, 0]
        else:
            outcome = paddle.mean(x_features, axis=1)
        outcome = self.head(outcome)
        return outcome


class VisionTransformer(PaddleVisionTransformerBase):
    """
    一个更完整的Vision Transformer实现，继承自基类。
    它增加了对全局平均池化（Global Average Pooling）的支持，作为CLS Token的替代方案。
    """
    def __init__(self, global_pool=False, **kwargs):
        self.actual_num_classes = kwargs.pop("num_classes", 1000)
        num_classes_for_base = 0 if global_pool else self.actual_num_classes
        super().__init__(num_classes=num_classes_for_base, **kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            _norm_layer_for_fc = kwargs.get(
                "norm_layer",
                (
                    self._norm_layer
                    if hasattr(self, "_norm_layer") and callable(self._norm_layer)
                    else partial(nn.LayerNorm, epsilon=1e-6)
                ),
            )
            self.fc_norm = _norm_layer_for_fc(self.embed_dim)
            if self.actual_num_classes > 0:
                self.head_after_gap = nn.Linear(self.embed_dim, self.actual_num_classes)
            else:
                self.head_after_gap = nn.Identity()

    def forward(self, x):
        """
        根据是否使用全局池化，执行相应的前向传播逻辑。
        """
        all_token_features = self.forward_features(x)
        if self.global_pool:
            start_idx = 1 if self.cls_token is not None else 0
            patch_tokens = all_token_features[:, start_idx:, :]
            pooled_features = paddle.mean(patch_tokens, axis=1)
            outcome = self.fc_norm(pooled_features)
            outcome = self.head_after_gap(outcome)
        else:
            if self.cls_token is not None:
                cls_token_feature = all_token_features[:, 0]
            else:
                cls_token_feature = paddle.mean(all_token_features, axis=1)
            outcome = self.head(cls_token_feature)
        return outcome


def RETFound_mae(**kwargs):
    """
    RETFound_mae模型的工厂函数。
    此函数用于实例化一个符合RETFound论文描述的ViT-Large架构的模型。
    
    Args:
        **kwargs: 传递给VisionTransformer构造函数的额外参数，用于覆盖默认配置。

    Returns:
        VisionTransformer: 一个配置好的ViT-Large模型实例。
    """
    default_vit_params = {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "patch_size": 16,
        "img_size": 224,
        "in_chans": 3,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "use_cls_token": True,
        "global_pool": False,
    }
    final_vit_params = default_vit_params.copy()
    final_vit_params.update(kwargs)
    norm_layer_arg = final_vit_params.pop("norm_layer", None)
    if norm_layer_arg is None:
        norm_layer_arg = partial(nn.LayerNorm, epsilon=1e-6)
    final_vit_params["norm_layer"] = norm_layer_arg
    print(f"Instantiating VisionTransformer (for RETFound_mae) with args: {final_vit_params}")
    model = VisionTransformer(**final_vit_params)
    return model


def RETFound_dinov2(**kwargs):
    """
    RETFound_dinov2模型的工厂函数（占位符）。
    用于加载或实现DINOv2模型的PaddlePaddle版本。
    """
    paddle_model_func_name = "vit_large_patch14_dinov2_lvd142m"
    print(f"Attempting to find or load DINOv2 model: {paddle_model_func_name}")
    if hasattr(paddle.vision.models, paddle_model_func_name):
        model_func = getattr(paddle.vision.models, paddle_model_func_name)
        model_paddle_kwargs = {
            "pretrained": kwargs.pop("pretrained", True),
            "num_classes": kwargs.pop("num_classes", 0),
        }
        if "img_size" in kwargs:
            model_paddle_kwargs["image_size"] = kwargs.pop("img_size")
        model_paddle_kwargs.update(kwargs)
        model = model_func(**model_paddle_kwargs)
    else:
        raise NotImplementedError(
            f"PaddlePaddle model '{paddle_model_func_name}' (for DINOv2) not found in paddle.vision.models. "
            "Manual implementation or weight conversion for this DINOv2 model is required."
        )
    return model