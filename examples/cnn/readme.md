# CNN（MNIST）

CNN 算法在 **MNIST 手写数字数据集** 上的最小可运行示例。

- **目标**：提供一个**可直接运行**的传统机器学习 / 浅层神经网络基线  
- **定位**：项目示例（非完整教程）

📄 **结果分析**：  
实验结果与分析请见 [`docs/cnn/cnn_report_zh.md`](../docs/cnn/cnn_report_zh.md)

📘 **原理说明**：  
CNN 的原理与推导请见：[`docs/cnn/CNN.md`](../docs/cnn/CNN.md)

---

## 目录结构

```text
cnn/
├── model.py                # 模型定义
├── train.py                # 模型训练主程序
├── mnist_net.npz           # 已训练的模型数据（可以通过model.load_weights()导入）
├── test.py                 # 模型测试主程序，画图程序
└── requirements.txt        # 依赖列表
```
环境要求
`
Python ≥ 3.8
`

安装依赖：
```
pip install -r requirements.txt
```

在 cnn/ 目录下执行：
```
python train.py
```

程序将加载 MNIST 数据并开始训练。

执行：
```
python test.py
```

程序将启动测试脚本并画图。