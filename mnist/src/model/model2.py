import torch
import torch.nn as nn
from torch.nn import functional as F

class Expression(nn.Module):
    """Lambda層封裝（保持與原Model兼容）"""
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class DropBlock2D(nn.Module):
    """結構化隨機丟棄模塊（適用於卷積層）"""
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
        return x * (1 - mask_block)

class DropBlock1D(nn.Module):
    """結構化隨機丟棄模塊（專用於全連接層）"""
    def __init__(self, block_size=3, drop_prob=0.1):
        super(DropBlock1D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
            
        # 1D版本gamma計算
        gamma = (self.drop_prob / self.block_size) * (x.shape[-1] / 
                (x.shape[-1] - self.block_size + 1))
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask_block = F.max_pool1d(
            mask.unsqueeze(1),  # 轉為 [B,1,D]
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        ).squeeze(1)
        return x * (1 - mask_block)

class Model2(nn.Module):
    def __init__(self, i_c=1, n_c=10):
        super(Model2, self).__init__()
        
        # 第一階段：深度可分離卷積 + 殘差
        self.conv1 = nn.Sequential(
            nn.Conv2d(i_c, 16, 3, stride=1, padding=1, groups=1),
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
            DropBlock2D(block_size=3, drop_prob=0.1),  # 2D版本用於卷積層
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.shortcut = nn.Identity()  # 殘差捷徑
        
        # 分類頭
        self.flatten = Expression(lambda x: x.mean(dim=[2, 3]))  # 全局平均池化
        self.fc1 = nn.Linear(32, 256)
        self.drop = DropBlock1D(block_size=5, drop_prob=0.2)  # 1D版本用於全連接層
        self.fc2 = nn.Linear(256, n_c)

    def forward(self, x_i, _eval=False):
        """支持訓練/評估模式切換"""
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
        x_o = self.flatten(x_o)  # [B,32,14,14] -> [B,32]
        x_o = F.relu(self.fc1(x_o))  # [B,32] -> [B,256]
        x_o = self.drop(x_o)  # 結構化隨機丟棄（1D版本）
        
        self.train()  # 恢復訓練模式
        return self.fc2(x_o)  # [B,256] -> [B,10]

    def get_features(self, x):
        """特徵提取接口（輸出卷積後的特徵圖）"""
        x = self.conv1(x)
        x = self.depthwise(x)
        return self.pointwise(x)

if __name__ == '__main__':
    # 測試案例
    input_tensor = torch.randn(4, 1, 28, 28)  # 模擬MNIST輸入
    model = Model2()
    
    # 維度驗證
    print("輸入維度:", input_tensor.shape)  # torch.Size([4, 1, 28, 28])
    output = model(input_tensor)
    print("輸出維度:", output.shape)  # torch.Size([4, 10])
    
    # 模式切換測試
    print("\n訓練模式輸出（前5個元素）:")
    print(output[0, :5])
    print("\n評估模式輸出（前5個元素）:")
    print(model(input_tensor, _eval=True)[0, :5])

