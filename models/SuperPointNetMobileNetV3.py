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

class InvertedResidualV3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se=False, activation='relu'):
        super(InvertedResidualV3, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_se = use_se

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(Conv2DBNBlock(in_channels, hidden_dim, kernel_size=1, activation=activation))
        
        # Depthwise
        layers.append(Conv2DBNBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                                  stride=stride, activation=activation))
        
        if use_se:
            layers.append(SEBlock(hidden_dim))
        
        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.activation = HSwish() if activation == 'hswish' else nn.ReLU6(inplace=True)

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = result + x
        return result

class SuperPointNetMobileNetV3(nn.Module):
    def __init__(self):
        super(SuperPointNetMobileNetV3, self).__init__()
        
        # Initial conv layer
        self.conv1 = Conv2DBNBlock(1, 16, kernel_size=3, stride=2, activation='hswish')
        
        # MobileNetV3 backbone blocks
        self.stage1 = nn.Sequential(
            InvertedResidualV3(16, 16, kernel_size=3, stride=1, expand_ratio=1, use_se=True, activation='hswish'),
            InvertedResidualV3(16, 24, kernel_size=3, stride=2, expand_ratio=4.5, use_se=False, activation='hswish'),
            InvertedResidualV3(24, 24, kernel_size=3, stride=1, expand_ratio=3.67, use_se=True, activation='hswish')
        )
        
        self.stage2 = nn.Sequential(
            InvertedResidualV3(24, 40, kernel_size=5, stride=2, expand_ratio=4, use_se=True, activation='hswish'),
            InvertedResidualV3(40, 40, kernel_size=5, stride=1, expand_ratio=6, use_se=True, activation='hswish'),
            InvertedResidualV3(40, 40, kernel_size=5, stride=1, expand_ratio=6, use_se=True, activation='hswish')
        )
        
        self.stage3 = nn.Sequential(
            InvertedResidualV3(40, 80, kernel_size=3, stride=2, expand_ratio=6, use_se=False, activation='relu'),
            InvertedResidualV3(80, 80, kernel_size=3, stride=1, expand_ratio=2.5, use_se=False, activation='relu'),
            InvertedResidualV3(80, 80, kernel_size=3, stride=1, expand_ratio=2.3, use_se=False, activation='relu')
        )
        
        self.stage4 = nn.Sequential(
            InvertedResidualV3(80, 112, kernel_size=3, stride=1, expand_ratio=6, use_se=True, activation='relu'),
            InvertedResidualV3(112, 112, kernel_size=3, stride=1, expand_ratio=6, use_se=True, activation='relu'),
            InvertedResidualV3(112, 160, kernel_size=5, stride=2, expand_ratio=6, use_se=True, activation='relu'),
            InvertedResidualV3(160, 160, kernel_size=5, stride=1, expand_ratio=6, use_se=True, activation='relu'),
            InvertedResidualV3(160, 160, kernel_size=5, stride=1, expand_ratio=6, use_se=True, activation='relu')
        )

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

        # Detector Head
        semi = self.detector(x)

        # Descriptor Head
        desc = self.descriptor(x)
        desc = nn.functional.normalize(desc, p=2, dim=1)

        return semi, desc 