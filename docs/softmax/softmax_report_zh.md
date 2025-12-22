# Softmax 模型训练与评估报告

## 1 模型与实现

### 1.1 Softmax 数值稳定性问题

Softmax 函数的标准形式为 $S(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$，其中 $z_i$ 是模型的原始输出分数（logits）。

该形式在计算上存在一个严重缺陷：**数值溢出 (Overflow)**。在深度学习中，模型的原始输出 $z_i$ 可能非常大（例如 1000）或非常小（例如 -1000）。如果 $z_i$ 中的值很大，$e^{1000}$ 将是一个超出计算机浮点数表示范围的巨大数字，导致计算结果变为 $inf$ (无穷大)，后续的除法 $inf / inf$ 将得到 $NaN$ (Not a Number)，导致计算失败。

为了解决这个问题，我们利用一个数学技巧：在分子和分母上同乘一个常数 $C$。

$S(z_i) = \frac{Ce^{z_i}}{\sum_{j} Ce^{z_j}} = \frac{e^{z_i + \log C}}{\sum_{j} e^{z_j + \log C}}$

我们巧妙地选择 $C = e^{-z_{\max}}$，其中 $z_{\max} = \max(z_j)$ 是所有 $z$ 中的最大值。将 $\log C = -z_{\max}$ 代入，公式变为：

$S(z_i) = \frac{e^{z_i - z_{\max}}}{\sum_{j} e^{z_j - z_{\max}}}$

这个“减去最大值”的技巧确保了指数函数的最大输入为 $z_{\max} - z_{\max} = 0$，因此其输出 $e^0 = 1$。所有其他项 $e^{z_j - z_{\max}}$ 都是 $e$ 的负数次幂（或 0），其值域稳定在 $(0, 1]$ 之间。这有效避免了上溢出问题，使得 Softmax 的计算在数值上保持稳定。此方法已在 `softmax.py` 脚本的 `softmax` 函数中实现。

### 1.2 数据加载与预处理代码说明：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

X_train_full = train_data.data.numpy().reshape(-1, 28*28).astype("float32") / 255.0
y_train_full = train_data.targets.numpy()
X_test = test_data.data.numpy().reshape(-1, 28*28).astype("float32") / 255.0
y_test = test_data.targets.numpy()

X_val, y_val = X_train_full[50000:], y_train_full[50000:]
X_train, y_train = X_train_full[:50000], y_train_full[:50000]

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")
```

这段代码负责加载和准备 MNIST 数据集：

1.  **数据加载**：使用 `torchvision.datasets.MNIST` 下载并加载 MNIST 数据集。
2.  **手动预处理**：
      * `train_data.data.numpy()`：提取原始数据（像素值范围 $0$ - $255$）并转换为 NumPy 数组。
      * `.reshape(-1, 28*28)`：将每张 $28 \times 28$ 像素的图像“展平”为一个包含 784 个元素的一维向量。
      * `.astype("float32") / 255.0`：将数据类型转为浮点数，并将像素值从 $[0, 255]$ 区间归一化到 $[0.0, 1.0]$ 区间。这是模型实际使用的输入数据。
3.  **数据集划分**：
      * 原始的 60000 个训练样本被分为两部分：
      * `X_train`, `y_train`：前 50000 个样本作为训练集。
      * `X_val`, `y_val`：后 10000 个样本作为验证集，用于在训练过程中监控模型性能。
      * `X_test`, `y_test`：10000 个独立的测试集样本，用于最终的模型评估。

### 1.3 模型初始化与训练代码说明：

```python
# 1. 初始化模型
clf = NumpySoftmax(input_dim=28*28, num_classes=10, lr=0.1, reg=1e-4)

# 2. 训练模型 (使用100个epoch进行演示)
history = clf.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=256)
```

这部分是模型训练的核心：

1.  **初始化模型**：
      * `clf = NumpySoftmax(...)`：创建了 `NumpySoftmax` 类的一个实例。
      * `input_dim=28*28`：指定输入维度为 784，对应展平后的图像向量。
      * `num_classes=10`：指定输出类别为 10 (数字 0 到 9)。
      * `lr=0.1`：设置学习率 (Learning Rate) 为 0.1，这是梯度下降中更新权重的步长。
      * `reg=1e-4`：设置 L2 正则化强度为 0.0001，用于防止模型过拟合。
2.  **训练模型**：
      * `clf.train(...)`：调用模型的训练方法。
      * `X_train, y_train`：传入训练集数据和标签。
      * `X_val, y_val`：传入验证集数据和标签，用于在每个轮次 (epoch) 结束时评估准确率。
      * `epochs=100`：模型将完整遍历训练数据集 100 次。
      * `batch_size=256`：在每个轮次中，数据被分成大小为 256 的小批量 (mini-batches) 进行处理和梯度更新。

## 2. 训练过程

![训练曲线](training_history.png)

* **左图 (训练损失曲线)**：显示了模型在训练集上的损失随轮次（Epoch）的变化。损失持续下降，表明模型在有效学习。
* **右图 (验证集准确率曲线)**：显示了模型在验证集上的准确率随轮次的变化。准确率稳步提升并趋于平稳，表明模型收敛良好。

## 3. 模型评估

### 3.1 总体准确率

模型在 MNIST 测试集上的最终准确率为： **0.9227**

### 3.2 详细分类报告

| 类别 | 精确率 (Precision) | 召回率 (Recall) | F1 分数 | 支持数 (Support) |
|:---|:---|:---|:---|:---|
| 0 | 0.9570 | 0.9776 | 0.9672 | 980 |
| 1 | 0.9661 | 0.9789 | 0.9724 | 1135 |
| 2 | 0.9270 | 0.8983 | 0.9124 | 1032 |
| 3 | 0.9030 | 0.9129 | 0.9079 | 1010 |
| 4 | 0.9286 | 0.9267 | 0.9276 | 982 |
| 5 | 0.8936 | 0.8666 | 0.8799 | 892 |
| 6 | 0.9351 | 0.9478 | 0.9414 | 958 |
| 7 | 0.9259 | 0.9232 | 0.9245 | 1028 |
| 8 | 0.8851 | 0.8778 | 0.8814 | 974 |
| 9 | 0.8952 | 0.9058 | 0.9005 | 1009 |

| **Accuracy** | | | **0.9227** | **10000** |
| Macro Avg | 0.9217 | 0.9215 | 0.9215 | 10000 |
| Weighted Avg | 0.9225 | 0.9227 | 0.9225 | 10000 |

### 3.3 混淆矩阵

![混淆矩阵](confusion_matrix.png)

* 混淆矩阵展示了模型预测的详细情况。
* **行（Y轴）** 代表真实的标签。
* **列（X轴）** 代表模型预测的标签。
* 对角线上的数字表示预测正确的样本数量，颜色越深代表数量越多。
* 非对角线上的数字（例如第4行第9列）表示有多少个真实的“4”被错误地预测为了“9”。

## 4. 模型洞察

### 4.1 学习到的权重（模板）

![权重可视化](weights.png)

* 上图展示了模型为 10 个类别（数字 0 到 9）分别学习到的权重 $W$。
* 每一张 $28 \times 28$ 的图像代表一个类别的“模板”。
* 高亮（颜色较亮）的区域表示模型认为对判断该数字**最重要**的像素。例如，数字“0”的模板在边缘是亮的，中间是暗的。


### 4.2 错误样本分析

![错误样本](misclassified.png)

* 上图随机抽取了 10 个模型在测试集上预测错误的样本。
* 每个样本下方标注了其 **真实标签** 和模型的 **错误预测**。
* 这有助于我们直观地理解模型容易在哪些“模棱两可”的（或书写不清的）数字上犯错。
