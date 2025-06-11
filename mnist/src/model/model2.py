import torch
import torch.nn as nn
from torch.nn import functional as F

class Expression(nn.Module):
    """保持與原Model相同的lambda層封裝"""
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class DropBlock2D(nn.Module):
    """新增結構化隨機丟棄模塊（對抗魯棒性關鍵）"""
    def __init__(self, block_size=3, drop_prob=0.1):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
            
        # 計算gamma值（公式來自原論文）
        gamma = (self.drop_prob / (self.block_size ** 2)) * (x.shape[-1] ** 2) / \
                ((x.shape[-1] - self.block_size + 1) ** 2)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask_block = F.max_pool2d(
            mask, 
            kernel_size=self.block_size, 
            stride=1, 
            padding=self.block_size // 2
        )
        x = x * (1 - mask_block)
        return x

class Model2(nn.Module):
    def __init__(self, i_c=1, n_c=10):
        super(Model2, self).__init__()
        
        # 第一階段：深度可分離卷積 + 殘差
        self.conv1 = nn.Sequential(
            nn.Conv2d(i_c, 16, 3, stride=1, padding=1, groups=1),  # 初始分組=1
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.depthwise = nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16)  # 步長2替代池化
        self.pointwise = nn.Conv2d(16, 32, 1)  # 通道擴展
        
        # 第二階段：殘差塊
        self.resblock = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DropBlock2D(block_size=3),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.shortcut = nn.Identity()  # 殘差捷徑
        
        # 分類頭
        self.flatten = Expression(lambda tensor: tensor.mean(dim=[2, 3]))  # 全局平均池化
        self.fc1 = nn.Linear(32, 256)
        self.drop = DropBlock2D()  # 用於全連接層的1D化版本
        self.fc2 = nn.Linear(256, n_c)

    def forward(self, x_i, _eval=False):
        """保持與原Model相同的_eval模式切換"""
        if _eval:
            self.eval()
        else:
            self.train()
            
        # 主幹網絡
        x_o = self.conv1(x_i)
        x_o = self.depthwise(x_o)
        x_o = self.pointwise(x_o)
        
        # 殘差連接
        residual = self.shortcut(x_o)
        x_o = self.resblock(x_o)
        x_o += residual
        x_o = F.relu(x_o)
        
        # 分類頭
        x_o = self.flatten(x_o)
        x_o = F.relu(self.fc1(x_o))
        x_o = self.drop(x_o)  # 結構化隨機丟棄
        
        self.train()  # 恢復訓練模式
        return self.fc2(x_o)

    def get_features(self, x):
        """新增特徵提取接口（用於梯度正交化計算）"""
        x = self.conv1(x)
        x = self.depthwise(x)
        return self.pointwise(x)

if __name__ == '__main__':
    # 測試輸出維度（與原Model相同格式）
    i = torch.FloatTensor(4, 1, 28, 28)
    n = Model2()
    print(n(i).size())  # 預期輸出: torch.Size([4, 10])
