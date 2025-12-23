import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import nailorch
from nailorch import optimizers
from nailorch import DataLoader
from nailorch.datasets import MNIST
from model import MNISTNet

def train():
    # 0. 超参数
    max_epoch = 20
    batch_size = 100
    lr = 0.001

    # 1. 准备数据
    # 注意：这里去掉了不支持的 flatten 参数
    train_set = MNIST(train=True)
    test_set = MNIST(train=False)
    
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    # 2. 模型与优化器
    model = MNISTNet()
    optimizer = optimizers.Adam(alpha=lr).setup(model)

    if nailorch.cuda.gpu_enable:
        model.to_gpu()

    # 记录训练历史
    history = {'train_loss': [], 'test_acc': []}

    # 3. 训练循环
    for epoch in range(max_epoch):
        start_time = time.time()
        sum_loss = 0
        
        # === Training Phase ===
        for x, t in train_loader:
            if nailorch.cuda.gpu_enable:
                x = nailorch.cuda.as_cupy(x)
                t = nailorch.cuda.as_cupy(t)
            
            # 【关键修复】将 (N, 784) 重塑为 (N, 1, 28, 28) 以适应卷积层
            x = x.reshape(-1, 1, 28, 28)
                
            y = model(x)
            loss = nailorch.functions.softmax_cross_entropy(y, t)
            
            model.cleargrads()
            loss.backward()
            optimizer.update()
            
            sum_loss += float(loss.data) * len(t)

        avg_loss = sum_loss / len(train_set)
        
        # === Validation Phase ===
        with nailorch.no_grad():
            correct_count = 0
            for x, t in test_loader:
                if nailorch.cuda.gpu_enable:
                    x = nailorch.cuda.as_cupy(x)
                    t = nailorch.cuda.as_cupy(t)
                
                # 【关键修复】验证时同样需要重塑形状
                x = x.reshape(-1, 1, 28, 28)

                y = model(x)
                pred = y.data.argmax(axis=1)
                correct_count += (pred == t).sum()
            
            if nailorch.cuda.gpu_enable:
                import cupy
                correct_count = cupy.asnumpy(correct_count)
                
            test_acc = float(correct_count) / len(test_set)

        elapsed = time.time() - start_time
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed:.2f}s')
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)

    # 4. 保存
    model.save_weights('mnist_net.npz')
    np.save('history.npy', history)
    print("训练完成！模型已保存为 mnist_net.npz")

if __name__ == '__main__':
    train()