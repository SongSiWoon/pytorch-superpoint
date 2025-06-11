import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3, inplace=self.inplace) / 6

class Conv2DBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bias=False, activation='relu'):
        super(Conv2DBNBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = HSwish() if activation == 'hswish' else nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 use_multi_query=False, query_h_strides=1, query_w_strides=1, kv_strides=1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_multi_query = use_multi_query
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides

        if use_multi_query:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim // num_heads, bias=qkv_bias)
            self.v = nn.Linear(dim, dim // num_heads, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C

        if self.use_multi_query:
            # Multi-query attention
            q = self.q(x).reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Downsample key and value
            k = x[:, ::self.kv_strides, ::self.kv_strides]
            v = x[:, ::self.kv_strides, ::self.kv_strides]
            k = self.k(k).reshape(B, -1, 1, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(v).reshape(B, -1, 1, self.head_dim).permute(0, 2, 1, 3)
        else:
            # Standard multi-head attention
            qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class UniversalInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size,
                 stride, expand_ratio, use_se=False, use_layer_scale=False, activation='relu'):
        super(UniversalInvertedBottleneck, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(Conv2DBNBlock(in_channels, hidden_dim, kernel_size=1, activation=activation))
        
        # First depthwise
        layers.append(Conv2DBNBlock(hidden_dim, hidden_dim, kernel_size=start_dw_kernel_size, 
                                  stride=stride, activation=activation))
        
        # Middle depthwise
        if middle_dw_kernel_size > 0:
            layers.append(Conv2DBNBlock(hidden_dim, hidden_dim, kernel_size=middle_dw_kernel_size, 
                                      stride=1, activation=activation))
        
        if use_se:
            layers.append(SEBlock(hidden_dim))
        
        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.activation = HSwish() if activation == 'hswish' else nn.ReLU6(inplace=True)
        
        if use_layer_scale:
            self.layer_scale = LayerScale(out_channels)
        else:
            self.layer_scale = None

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = result + x
        if self.layer_scale is not None:
            result = self.layer_scale(result)
        return result

class FusedInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, 
                 use_se=False, use_layer_scale=False, activation='relu'):
        super(FusedInvertedBottleneck, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Fused expansion
            layers.append(Conv2DBNBlock(in_channels, hidden_dim, kernel_size=kernel_size, 
                                      stride=stride, activation=activation))
        else:
            # Depthwise
            layers.append(Conv2DBNBlock(in_channels, hidden_dim, kernel_size=kernel_size, 
                                      stride=stride, activation=activation))
        
        if use_se:
            layers.append(SEBlock(hidden_dim))
        
        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.activation = HSwish() if activation == 'hswish' else nn.ReLU6(inplace=True)
        
        if use_layer_scale:
            self.layer_scale = LayerScale(out_channels)
        else:
            self.layer_scale = None

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = result + x
        if self.layer_scale is not None:
            result = self.layer_scale(result)
        return result

class SuperPointNetMobileNetV4(nn.Module):
    def __init__(self):
        super(SuperPointNetMobileNetV4, self).__init__()
        
        # Initial conv layer
        self.conv1 = Conv2DBNBlock(1, 16, kernel_size=3, stride=2, activation='hswish')
        
        # MobileNetV4 backbone blocks
        self.stage1 = nn.Sequential(
            FusedInvertedBottleneck(16, 16, kernel_size=3, stride=1, expand_ratio=1, 
                                  use_se=True, use_layer_scale=True, activation='hswish'),
            UniversalInvertedBottleneck(16, 24, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=2, expand_ratio=4.5, use_se=False, 
                                      use_layer_scale=True, activation='hswish'),
            UniversalInvertedBottleneck(24, 24, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=3.67, use_se=True, 
                                      use_layer_scale=True, activation='hswish')
        )
        
        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(24, 40, start_dw_kernel_size=5, middle_dw_kernel_size=0,
                                      stride=2, expand_ratio=4, use_se=True, 
                                      use_layer_scale=True, activation='hswish'),
            UniversalInvertedBottleneck(40, 40, start_dw_kernel_size=5, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='hswish'),
            UniversalInvertedBottleneck(40, 40, start_dw_kernel_size=5, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='hswish')
        )
        
        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(40, 80, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=2, expand_ratio=6, use_se=False, 
                                      use_layer_scale=True, activation='relu'),
            UniversalInvertedBottleneck(80, 80, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=2.5, use_se=False, 
                                      use_layer_scale=True, activation='relu'),
            UniversalInvertedBottleneck(80, 80, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=2.3, use_se=False, 
                                      use_layer_scale=True, activation='relu')
        )
        
        self.stage4 = nn.Sequential(
            UniversalInvertedBottleneck(80, 112, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='relu'),
            UniversalInvertedBottleneck(112, 112, start_dw_kernel_size=3, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='relu'),
            UniversalInvertedBottleneck(112, 160, start_dw_kernel_size=5, middle_dw_kernel_size=0,
                                      stride=2, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='relu'),
            UniversalInvertedBottleneck(160, 160, start_dw_kernel_size=5, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='relu'),
            UniversalInvertedBottleneck(160, 160, start_dw_kernel_size=5, middle_dw_kernel_size=0,
                                      stride=1, expand_ratio=6, use_se=True, 
                                      use_layer_scale=True, activation='relu')
        )

        # Attention blocks
        self.attention1 = MultiHeadSelfAttention(160, num_heads=8, use_multi_query=True,
                                               query_h_strides=2, query_w_strides=2, kv_strides=2)
        self.attention2 = MultiHeadSelfAttention(160, num_heads=8, use_multi_query=True,
                                               query_h_strides=2, query_w_strides=2, kv_strides=2)

        # Detector Head
        self.detector = nn.Sequential(
            Conv2DBNBlock(160, 256, kernel_size=3, activation='hswish'),
            nn.Conv2d(256, 65, kernel_size=1, bias=False),
            nn.BatchNorm2d(65)
        )

        # Descriptor Head
        self.descriptor = nn.Sequential(
            Conv2DBNBlock(160, 256, kernel_size=3, activation='hswish'),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        # Shared Encoder
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Attention
        x = x + self.attention1(x)
        x = x + self.attention2(x)

        # Detector Head
        semi = self.detector(x)

        # Descriptor Head
        desc = self.descriptor(x)
        desc = nn.functional.normalize(desc, p=2, dim=1)

        return semi, desc 