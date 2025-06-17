import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import os

class Conv2DBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bias=False):
        super(Conv2DBNBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(Conv2DBNBlock(in_channels, hidden_dim, kernel_size=1))
        
        # Depthwise
        layers.append(Conv2DBNBlock(hidden_dim, hidden_dim, kernel_size=3, stride=stride))
        
        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = result + x
        return result

class SuperPointNetMobileNetV2(nn.Module):
    def __init__(self, config):
        super(SuperPointNetMobileNetV2, self).__init__()
        
        # Initial conv layer - stride 1로 시작
        self.conv1 = Conv2DBNBlock(1, 32, kernel_size=3, stride=1)
        
        # MobileNetV2 backbone blocks - stride 조정
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=1),  # /2
            InvertedResidual(64, 64, stride=1, expand_ratio=6)
        )
        
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=6),  # /2
            InvertedResidual(128, 128, stride=1, expand_ratio=6)
        )
        
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 256, stride=2, expand_ratio=6),  # /2
            InvertedResidual(256, 256, stride=1, expand_ratio=6)
        )
        
        self.stage4 = nn.Sequential(
            InvertedResidual(256, 512, stride=1, expand_ratio=6),
            InvertedResidual(512, 512, stride=1, expand_ratio=6)
        )

        # Detector Head
        self.detector = nn.Sequential(
            Conv2DBNBlock(512, 256, kernel_size=3),
            nn.Conv2d(256, 65, kernel_size=1, bias=False),
            nn.BatchNorm2d(65)
        )

        # Descriptor Head
        self.descriptor = nn.Sequential(
            Conv2DBNBlock(512, 256, kernel_size=3),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )

    def forward(self, x, subpixel=False):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
          subpixel: Whether to use subpixel detection (not used in this implementation)
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # MobileNetV2 backbone
        x = self.conv1(x)  # /1
        x = self.stage1(x)  # /2
        x = self.stage2(x)  # /2
        x = self.stage3(x)  # /2
        x = self.stage4(x)  # /1
        
        # Detector Head
        semi = self.detector(x)
        
        # Descriptor Head
        desc = self.descriptor(x)
        
        # Normalize the descriptor
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        
        return semi, desc 

    def save_backbone(self, save_path):
        """백본 모델만 저장하는 메서드"""
        import os
        import torch
        
        # 백본 모델의 state dict 생성
        backbone_state = {
            'conv1': self.conv1.state_dict(),
            'stage1': self.stage1.state_dict(),
            'stage2': self.stage2.state_dict(),
            'stage3': self.stage3.state_dict(),
            'stage4': self.stage4.state_dict()
        }
        
        # 저장 경로 생성
        backbone_path = os.path.join(save_path, 'backbone.pth')
        
        # 저장
        torch.save(backbone_state, backbone_path)
        print(f"백본 모델이 {backbone_path}에 저장되었습니다.") 