import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """深度可分離卷積層，減少參數耦合"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=dilation, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.gelu(self.bn(x))

class InvertedResidualBlock(nn.Module):
    """反向殘差塊，擴張壓縮特徵維度"""
    def __init__(self, in_channels, expand_ratio=4, stride=1, drop_prob=0.1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1) and (in_channels == hidden_dim)
        
        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU()
                )
            )
        layers.extend([
            DepthwiseSeparableConv(hidden_dim, hidden_dim, stride=stride),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
            nn.GroupNorm(8, in_channels)
        ])
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(p=drop_prob) if drop_prob > 0 else nn.Identity()
        
    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        return self.dropout(self.conv(x))

class Model2(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.2):
        super().__init__()
        # 階段1: 高頻特徵提取 (與WideResNet的淺層正交)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            InvertedResidualBlock(32, expand_ratio=2, stride=2)
        )
        
        # 階段2: 多尺度特徵融合
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(32, expand_ratio=4, stride=1),
            InvertedResidualBlock(32, expand_ratio=2, stride=2, drop_prob=drop_rate)
        )
        
        # 階段3: 空洞卷積捕捉非局部模式
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1, dilation=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            InvertedResidualBlock(64, expand_ratio=2, stride=2)
        )
        
        # 輸出頭
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, _eval=False):
        if _eval:
            self.eval()
            
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        
        if _eval:
            self.train()
        return x

if __name__ == '__main__':
    model = Model2(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    print(model(x).shape)  # 輸出應為 [4, 10]
    print(f"參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
