import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import nailorch.functions as F
import nailorch.layers as L
from nailorch import Model

class MNISTNet(Model):
    def __init__(self):
        super().__init__()
        # 第一层卷积: 输出 16 通道, 3x3 核
        self.conv1 = L.Conv2d(16, kernel_size=3, stride=1, pad=1)
        # 第二层卷积: 输出 32 通道
        self.conv2 = L.Conv2d(32, kernel_size=3, stride=1, pad=1)
        # 全连接层 1
        self.fc1 = L.Linear(128)
        # 输出层 (10个数字)
        self.fc2 = L.Linear(10)

    def forward(self, x):
        # Layer 1: Conv -> ReLU -> MaxPool (28x28 -> 14x14)
        x = F.relu(self.conv1(x))
        x = F.pooling(x, 2, 2)
        
        # Layer 2: Conv -> ReLU -> MaxPool (14x14 -> 7x7)
        x = F.relu(self.conv2(x))
        x = F.pooling(x, 2, 2)
        
        # Flatten: 自动展平
        x = F.reshape(x, (x.shape[0], -1))
        
        # Layer 3: Linear -> ReLU
        x = F.relu(self.fc1(x))
        
        # Output
        x = self.fc2(x)
        return x