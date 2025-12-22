# TLNN

两层全连接神经网络在 MNIST 手写数字数据集上的最小可运行示例。

- 目标：提供一个**可直接运行**的传统机器学习基线
- 定位：项目示例（非完整教程）

📄 **结果分析**：
实验结果与分析请见 [`docs/tlnn/tlnn_report_zh.md`](../docs/tlnn/tlnn_report_zh.md)

📘 **原理说明**：  
详细原理与推导请见：[`docs/tlnn/TLNN.md`](../docs/KNN.md)

---

## 目录结构

```text
knn/
├── tlnn.py            # tlnn 实现与主程序
└── requirements.txt  # 依赖
```
MNIST数据集在主目录的data文件夹下

环境要求
`
Python ≥ 3.8
`

安装依赖：
```
pip install -r requirements.txt
```

在 tlnn/ 目录下执行：

python tlnn.py


程序将加载 MNIST 数据并输出分类结果（如准确率）。