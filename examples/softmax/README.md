# Softmax + MLP（MNIST）

Softmax + MLP 算法在 **MNIST 手写数字数据集** 上的最小可运行示例。

- **目标**：提供一个**可直接运行**的传统机器学习 / 浅层神经网络基线  
- **定位**：项目示例（非完整教程）

📄 **结果分析**：  
实验结果与分析请见 [`docs/softmax/softmax_report_zh.md`](../docs/softmax/softmax_report_zh.md)

📘 **原理说明**：  
Softmax 与 MLP 的原理与推导请见：[`docs/softmax/softmax.md`](../docs/softmax/softmax.md)

---

## 目录结构

```text
softmax/
├── softmax.py              # Softmax + MLP 实现与主程序
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

在 softmax/ 目录下执行：
```
python softmax.py
```


程序将加载 MNIST 数据并输出分类结果（如准确率）。