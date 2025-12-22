# KNN

KNN（k-Nearest Neighbors）算法在 MNIST 手写数字数据集上的最小可运行示例。

- 目标：提供一个**可直接运行**的传统机器学习基线
- 定位：项目示例（非完整教程）

详细原理与推导请见：[`docs/knn/KNN.md`](../docs/KNN.md)

---

## 目录结构

```text
knn/
├── KNN.py            # KNN 实现与主程序
├── mnist.json.gz     # MNIST 数据
└── requirements.txt  # 依赖
```

环境要求
`
Python ≥ 3.8
`

安装依赖：
```
pip install -r requirements.txt
```

在 knn/ 目录下执行：

python KNN.py


程序将加载 MNIST 数据并输出分类结果（如准确率）。