# 山东大学（威海）23级数科班大三上大作业第一阶段：用纯python实现KNN

## 目录

* [第一节 · KNN 原理与三种实现方法](#第一节--knn-原理与三种实现方法)

  * [1.1 KNN 基本思想](#11-knn-基本思想)
  * [1.2 距离度量](#12-距离度量)
  * [1.3 K 的选择与特征预处理](#13-k-的选择与特征预处理)
  * [1.4 复杂度与瓶颈](#14-复杂度与瓶颈)
  * [1.5 三种实现方法](#15-三种实现方法)

    * [1.5.1 KDTree与BallTree](#151-kdtree与balltree)
    * [1.5.2 Annoy（近似最近邻 · 多随机投影树）](#152-annoy近似最近邻--多随机投影树)
    * [1.5.3 HNSW（近似最近邻 · 分层可导航小世界图）](#153-hnsw近似最近邻--分层可导航小世界图)
  * [1.6 几种方法对比与选择](#16-几种方法对比与选择)
* [第二节 · 具体代码实现（KNN 手搓实践-python）](#第二节--具体代码实现knn-手搓实践-python)

  * [2.1 项目简介与效果预期](#21-项目简介与效果预期)
  * [2.2 环境准备](#22-环境准备)
  * [2.3 数据准备（mnist.json.gz）](#23-数据准备mnistjsongz)
  * [2.4 代码结构与运行方式](#24-代码结构与运行方式)

    * [2.4.1 目录结构](#241-目录结构)
    * [2.4.2 直接运行（CuPy 版，GPU）](#242-直接运行cupy-版gpu)
  * [2.5 KNN 代码逐段讲解](#25-knn-代码逐段讲解)

    * [2.5.1 数据读取与预处理](#251-数据读取与预处理)
    * [2.5.2 KNN 核心类](#252-knn-核心类)
    * [2.5.3 用验证集选择最优 k](#253-用验证集选择最优-k)
    * [2.5.4 计算在测试集上的最终结果](#254-计算在测试集上的最终结果)
    * [2.5.5 可视化前若干预测结果](#255-可视化前若干预测结果)
  * [2.6 指标与可视化](#26-指标与可视化)

    * [2.6.1 最优k与准确率](#261-最优k与准确率)
    * [2.6.2 混淆矩阵](#262-混淆矩阵)
 * [附：KNN.py](#附knnpy)

小组成员：魏子钦，刘艺航，钦浩然，刘宇，李暄


# 第一节 · KNN 原理与三种实现方法

> 前言：本节主要讲述KNN 的核心思想、距离度量、K 的选择、复杂度瓶颈，以及三种常见实现/加速方案：**KDTree**、**Annoy**、**HNSW**。附带简洁的可运行示例，便于读者上手理解。


## 1.1 KNN 基本思想

**K 近邻（K-Nearest Neighbors), KNN** 是基于实例的监督学习方法。给定一个新样本，计算其与训练集中所有样本的距离，选出最近的 **K 个邻居**，用**多数投票**或**平均**得到输出。它的基本思想是从训练集中寻找和输入样本最相似的k个样本，如果这k个样本中的大多数属于某一个类别，则输入的样本也属于这个类别。关于KNN算法，一个核心问题是：如何快速从数据集中找到和目标样本最接近的K个样本？


## 1.2 距离度量

设两个样本
$
x=(x_1,\dots,x_d),\quad y=(y_1,\dots,y_d)
$
大多数情况下，我们选择**欧氏距离**：
$
d(x,y)=\sqrt{\sum_{i=1}^{d}(x_i-y_i)^2}
$

除此之外，有个别情况会用到如下距离：

* **L1 / 曼哈顿距离**：$\sum |x_i-y_i|$
* **闵可夫斯基距离**：$(\sum |x_i-y_i|^p)^{1/p}$
* **余弦相似度**：$\frac{x\cdot y}{|x||y|}$



## 1.3 K 的选择与特征预处理

* **K 太小** → 对噪声敏感（过拟合）；**K 太大** → 决策边界过于平滑（欠拟合）。
* 实务中**用验证集/交叉验证**选 K。
* **特征归一化/标准化是必要的**，避免某些量纲支配距离：

  * 归一化：$(x-\min)/( \max-\min)$
  * 标准化：$(x-\mu)/\sigma$


## 1.4 复杂度与瓶颈

* 朴素暴力搜索（Brute-Force）对每个测试样本要与**全部训练样本**算一次距离 →
  **时间复杂度** ($\mathcal{O}(N_{\text{train}}\cdot d)$)（每个测试样本），整体 $\mathcal{O}(N_{\text{test}}N_{\text{train}}d)$。
* 高维（如 MNIST 784 维）与大样本下，**预测很慢**。解决思路：

  1. **空间索引（KDTree/BallTree）**：精确加速（低维有效）；
  2. **近似最近邻（Annoy/HNSW/FAISS 等）**：牺牲极少准确率换取巨大速度与扩展性。


## 1.5 三种实现方法

### 1.5.1 KDTree与BallTree

**原理**

kd 树是一种对k维特征空间中的实例点进行存储以便对其快速检索的树形数据结构。它是二叉树，核心思想是对 k 维特征空间不断切分（假设特征维度是768，对于(0,1,2,...,767)中的每一个维度，以中值递归切分）构造的树，每一个节点是一个超矩形，小于结点的样本划分到左子树，大于结点的样本划分到右子树。树构造完毕后，最终检索时：

（1）从根结点出发，递归地向下访问kd树。若目标点 当前维的坐标小于切分点的坐标，移动到左子树，否则移动到右子树，直至到达叶结点；

（2）以此叶结点为“最近点”，递归地向上回退，查找该结点的兄弟结点中是否存在更近的点，若存在则更新“最近点”，否则回退；未到达根结点时继续执行（2）；

（3）回退到根结点时，搜索结束。

kd树在维数小于20时效率最高，一般适用于训练实例数远大于空间维数时的k近邻搜索；当空间维数接近训练实例数时，它的效率会迅速下降，几乎接近线形扫描

**BallTree**

为了解决kd树在样本特征维度很高时效率低下的问题，研究人员提出了“球树“BallTree。KD 树沿坐标轴分割数据，BallTree将在一系列嵌套的超球面上分割数据，即使用超球面而不是超矩形划分区域。具体而言，BallTree 将数据递归地划分到由质心 C 和 半径 r 定义的节点上，以使得节点内的每个点都位于由质心C和半径 r 定义的超球面内。通过使用三角不等式$|X + Y| \le |X| + |Y|$减少近邻搜索的候选点数。

**示例-sklearn.neighbors.KDTree**

```python
import numpy as np
from sklearn.neighbors import KDTree
X = np.random.rand(1000, 10)  # 10维，低维更适用
tree = KDTree(X, leaf_size=40)  # 叶子节点容量可调
dist, ind = tree.query(np.random.rand(1, 10), k=5)  # 查询5近邻
print("最近邻索引:", ind, "距离:", dist)
```
最近邻索引: [[597 974 828  55  32]] 距离: [[0.39419227 0.43613598 0.45990816 0.5452833  0.56281512]]

**sklearn KNeighborsClassifier**
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### 1.5.2 Annoy（近似最近邻 · 多随机投影树）

**原理**

Annoy 全称 “Approximate Nearest Neighbors Oh Yeah”，是一种适合实际应用的快速相似查找算法。  
Annoy 同样通过建立一个二叉树来使得每个点查找时间复杂度是 $O(\log n)$，和 KD 树不同的是，Annoy 没有对 k 维特征进行切分。Annoy 的每一次空间划分，可以看作聚类数为 2 的 KMeans 过程。  
收敛后在产生的两个聚类中心连线之间建立一条垂线（图中的黑线），把数据空间划分为两部分。

<p align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/9a14d178f1d2471c91400f64bd4a4eca28ebc16e7f3949b494606f3200572a5a" width="420"><br>
</p>

在划分的子空间内不停的递归迭代继续划分，直到每个子空间最多只剩下 K 个数据节点，划分结束。

<p align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/1063c1db303c42c29178edb1048a2173a1416010167f47799f837660901df4a4" width="420"><br>
</p>

最终生成的二叉树具有如下类似结构，二叉树底层是叶子节点记录原始数据节点，其他中间节点记录的是分割超平面的信息。

<p align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/d18036aaed7a4adf86ea9f63a8a3323d9d0039ad382b4f6cb0d24169a38ca8d5" width="420"><br>
</p>

查询过程和 KD 树类似，先从根向叶子结点递归查找，再向上回溯即可。  

**示例**
```python
import numpy as np
from annoy import AnnoyIndex

f = 64  # 向量维度
index = AnnoyIndex(f, metric='euclidean')  # 'angular' 用于余弦
for i in range(10000):
    index.add_item(i, np.random.randn(f))
index.build(20)  # 建20棵树，越多越准但越慢

I, D = index.get_nns_by_vector(np.random.randn(f), 10, include_distances=True)
print("近邻索引:", I)
print("近邻距离:", D)
```

### 1.5.3 HNSW（近似最近邻 · 分层可导航小世界图）

**原理**

和前几种算法不同，HNSW（Hierarchcal Navigable Small World graphs）是基于图存储的数据结构。

朴素查找法：不少人脑子里都冒出过这样的朴素想法，把某些点和点之间连上线，构成一个查找图，存储备用；当我想查找与粉色点最近的一点时，我从任意一个黑色点出发，计算它和粉色点的距离，与这个任意黑色点有连接关系的点我们称之为“友点”，然后我要计算这个黑色点的所有“友点”与粉色点的距离，从所有“友点”中选出与粉色点最近的一个点，把这个点作为下一个进入点，继续按照上面的步骤查找下去。如果当前黑色点对粉色点的距离比所有“友点”都近，终止查找，这个黑色点就是我们要找的离粉色点最近的点。

HNSW算法就是对上述朴素思想的改进和优化。为了达到快速搜索的目标，hnsw算法在构建图时还至少要满足如下要求：

1）图中每个点都有“友点”；

2）相近的点都互为“友点”；

3）图中所有连线的数量最少；

4）配有高速公路机制的构图法。

HNSW低配版NSW论文中配了这样一张图，短黑线是近邻点连线，长红线是“高速公路机制”，如此可以大幅减少平均搜索的路径长度。

<p align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/d88a8857ecef4700bc6d080f90a9b634779ca2fc554e44e08d9c294e5f657e07" width="420"><br>
</p>

在NSW基础之上，HNSW加入了跳表结构做了进一步优化。最底层是所有数据点，每一个点都有50%概率进入上一层的有序链表。这样可以保证表层是“高速通道”，底层是精细查找。通过层状结构，将边按特征半径进行分层，使每个顶点在所有层中平均度数变为常数，从而将NSW的计算复杂度由多重对数复杂度降到了对数复杂度。

**示例（hnswlib）**
```python
import numpy as np
import hnswlib

dim = 64
num = 10000
data = np.float32(np.random.random((num, dim)))

index = hnswlib.Index(space='l2', dim=dim)  # 'cosine' 支持余弦
index.init_index(max_elements=num, ef_construction=200, M=16)  # M：每点连边数；ef_construction：建图搜索宽度
index.add_items(data, ids=np.arange(num))

index.set_ef(64)  # 查询时的探索宽度，越大越准（也越慢）
labels, dists = index.knn_query(data[:1], k=5)
print(labels, dists)
```

## 1.6 几种方法对比与选型建议

| 方法              | 精确/近似 | 查询速度   | 维度适配           | 内存      | 动态更新 | 典型场景         |
| --------------- | ----- | ------ | -------------- | ------- | ---- | ------------ |
| **Brute-Force** | 精确    | 慢（线性）  | 任意（高维也可）       | 低       | 易    | 小数据/教学/基准    |
| **KDTree**      | 精确    | 快（低维）  | 低维友好（<~20） | 低-中     | 一般   | 物理/地理/低维特征   |
| **Annoy**       | 近似    | 很快 | 高维可用           | 低   | 一般   | 大规模检索、推荐     |
| **HNSW**        | 近似    | 极快 | 高维最友好     | 中-高 | 好    | 向量数据库/NLP/图像 |

>在本文中我们只介绍了几种常用的k近邻查找算法，kdtree是KNN的一种基本实现算法；考虑到并发、延时等要素，annoy、hnsw是可以在实际业务中落地的算法，其中bert/sentence-bert+hnsw的组合会有不错的召回效果

# 第二章 · 具体代码实现（KNN 手搓实践-python）

> 前言：本部分包含环境准备、数据放置、代码工作原理、指标与可视化、优化建议等，后续会整理到最终大作业开源教材中，文档仍存在许多不足之处，欢迎大家批评指正！

## 2.1 项目简介与效果预期

本项目实现了K 近邻（KNN）分类器，对 **MNIST 手写数字**进行识别。思路非常直接：对每一张测试图片，在训练集中找到距离最近的 `k` 张图像，让这些邻居“投票”决定类别。本次作业中我们采用 **欧氏距离 + 多数投票**，在 MNIST 子集上获得**94.63**%的准确率。


## 2.2 环境准备

在新建项目后，建议选择**GPU**以便使用 **CuPy** 加速。如果你的环境暂时无法安装/使用 CuPy，仍可用 **CPU + NumPy** 跑通。

> 依赖清单见requirements.txt


## 2.3 数据准备（mnist.json.gz）

* 将 **`mnist.json.gz`** 上传到项目根目录，与 `KNN.py` **同级**。
* 该文件内包含 `train/val/test` 三个划分，每个划分由 `(X, y)` 构成；`X` 为图像数组，`y` 为标签。脚本会按此格式直接读取：

  ```python
  datafile = './mnist.json.gz'
  data = json.load(gzip.open(datafile))
  train_set, val_set, test_set = data
  X_train, y_train = train_set
  X_test,  y_test  = test_set
  ```

  > 因为脚本**直接解包出 train/val/test**，请确保文件结构与之匹配，否则会读取失败。


## 2.4 代码结构与运行方式

### 2.4.1 目录结构

```
.
├── KNN.py
└── mnist.json.gz
└── requirements.txt
```

### 2.4.2 直接运行（CuPy 版，GPU）

在终端中依次执行：

```bash
pip install -r requirements.txt
```

```bash
python KNN.py
```

## 2.5 KNN 代码逐段讲解

### 2.5.1 数据读取与预处理

* 从 `mnist.json.gz` 读入 `(train/val/test)`。（原始数据集已经预先划分好了，这里直接导入并解包）

  ```python
    data = json.load(gzip.open(datafile))
    train_set, val_set, test_set = data
    X_train, y_train = train_set
    X_test, y_test = test_set
    X_val, y_val = val_set
  ```

* 为了更快演示，脚本默认只取前 **10000** 个样本进行训练与测试（可自行调大/调小）：

* 将图像**展平、归一化到 [0,1]**：

  ```python
    X_train = np.array(X_train).reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test = np.array(X_test).reshape(len(X_test), -1).astype(np.float32) / 255.0
    y_train = np.array(y_train); y_test = np.array(y_test)
    X_val = np.array(X_val).reshape(len(X_val), -1).astype(np.float32) / 255.0
    y_val = np.array(y_val)
  ```

- 展平后维度从 (10000, 28, 28) 变为 (10000, 784)。现在每一行代表一个样本（一张图），每一列代表一个像素点。

> 子集训练是 KNN 的常见做法，因为 KNN 推理阶段复杂度随**训练集大小线性增长**。

### 2.5.2 KNN 核心类

```python
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):   # KNN 无需训练过程，仅保存数据,存的是train数据
        self.X = X
        self.y = y

    def predict(self, X):
        #这里的X是test或val
        preds = []
        for i, x in enumerate(X):
            dists  = np.sqrt(np.sum((self.X - x) ** 2, axis=1))     
            # 计算test或val中单条数据x与train中每一条数据（self.X) 的欧氏距离
            k_idx  = np.argsort(dists)[:self.k]                    
            # 对距离数组进行排序，找最近的 k 个邻居
            k_lbls = self.y[k_idx]
            # 利用这 k 个索引，从 `self.y` (所有训练标签) 中取出对应的标签。
            labels, counts = np.unique(k_lbls, return_counts=True)  
            # 统计这 k 个标签中，每个唯一标签出现的次数。
            pred = labels[np.argmax(counts)]
            # 用出现次数最多的标签作为预测结果
            preds.append(pred)
        return np.array(preds)
```

* **距离度量**：欧氏距离 $|x - x_i|_2 $
* **最近邻检索**：对所有距离**排序**并取前 `k`。
* **决策规则**：`k` 个邻居做**多数投票**。

### 2.5.3 用验证集选择最优 k

```python
def eval_k(k):
    knn_tmp = KNN(k=k)
    knn_tmp.fit(X_train, y_train)
    y_val_pred = knn_tmp.predict(X_val)
    acc_val = np.mean(y_val_pred == y_val)
    return float(acc_val)
candidates = [1,3,5,7,9,11,13,15]
scores = [(k, eval_k(k)) for k in candidates]
best_k, best_acc = max(scores, key=lambda x: x[1])
print("验证集最优 k:", best_k, "Acc:", f"{best_acc*100:.2f}%")
```

* 先实例化KNN类
* 然后输入不同的k值（ `k ∈ {1,3,5,7,9,11,13,15}`），输出在验证集上的准确率
* 准确率最高的k作为best_k用于测试集

### 2.5.4 计算在测试集上的最终结果

```python
# ========== 3. 训练与测试 ==========
print("训练 KNN（其实只是存数据到Model，这里我们象征性地fit一下）...")
# 将原来的固定 k=10 改为使用验证集挑选出来的 best_k
knn = KNN(k=best_k)
knn.fit(X_train, y_train)
print("预测中...")
y_pred = knn.predict(X_test)

# ========== 4. 计算准确率 ==========
acc = np.mean(y_pred == y_test)
print(f"测试集最终准确率: {acc * 100:.2f}%")
```

> 最终的结果为**94.63**%

![](https://ai-studio-static-online.cdn.bcebos.com/ceb6c790aa7f42a6a3b696294ef3832d8db07d828be049458997ae79287987b0)



### 2.5.5 可视化前若干预测结果

```python
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28).get(), cmap='gray')
    plt.title(f"预测:{y_pred[i]}, 真实:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```
![](https://ai-studio-static-online.cdn.bcebos.com/7038745935ec4c2cbff8368ba532b1391a1491271be64cd4a5d2e2ec36cb4144)


## 2.6 指标与可视化

### 2.6.1 最优k与准确率

 模型会自动在验证集上测试多个 k，选出准确率最高的最优 k，并输出验证集最优k值与验证集准确率，用这个k值给出测试集最终准确率

### 2.6.2 混淆矩阵

 直观看出哪些数字最易混淆：行表示 真实标签（True），列表示 预测标签（Predicted）。对角线越亮 → 模型预测正确越多。非对角线位置亮 → 代表易混类别。

```python
# --- Confusion Matrix（可选） ---
num_classes = int(np.max(y_test)) + 1
cm = np.zeros((num_classes, num_classes), dtype=np.int64)
for t, p in zip(y_test, y_pred):
    cm[int(t), int(p)] += 1

plt.figure(figsize=(6,5))
plt.imshow(cm if not hasattr(cm, "get") else cm.get(), interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.show()
```

![](https://ai-studio-static-online.cdn.bcebos.com/52e4568c5f8b4afb98be017ae291fad1e147df481a4c4c129459a4e16a3944a4)



# 附：KNN.py
```python
import matplotlib.pyplot as plt
import cupy as np
# 假如没有gpu的话，使用numpy
import json
import gzip
# ========== 1. 读取 MNIST ==========
datafile = './mnist.json.gz'
print('正在从 {} 中加载数据......'.format(datafile))
# 加载json数据文件
data = json.load(gzip.open(datafile))
print('Mnist 数据集加载完成')
# 读取到的数据区分训练集，验证集，测试集
train_set, val_set, test_set = data
X_train, y_train = train_set
X_test, y_test = test_set
X_val, y_val = val_set
# 为了演示快一点，先只用一小部分
index = 10000
X_train, y_train = X_train[:index], y_train[:index]
X_test, y_test = X_test[:index], y_test[:index]
X_val, y_val = X_val[:index], y_val[:index]

# 展平并归一化
X_train = np.array(X_train).reshape(len(X_train), -1).astype(np.float32) / 255.0
X_test = np.array(X_test).reshape(len(X_test), -1).astype(np.float32) / 255.0
y_train = np.array(y_train); y_test = np.array(y_test)
X_val = np.array(X_val).reshape(len(X_val), -1).astype(np.float32) / 255.0
y_val = np.array(y_val)
# ========== 2. 实现纯 NumPy KNN ==========
class KNN:
    def __init__(self, k): # 分类维度
        self.k = k
    def fit(self, X, y): # KNN不需要训练
        self.X = X
        self.y = y
    def predict(self, X):
        preds = []
        for i, x in enumerate(X):
            # 欧氏距离：sqrt(sum((x - X_train)^2))
            dists = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            # 取前 k 个最近邻
            k_idx = np.argsort(dists)[:self.k]
            k_labels = self.y[k_idx]
            # 投票：出现次数最多的标签
            labels, counts = np.unique(k_labels, return_counts=True)
            pred = labels[np.argmax(counts)]
            preds.append(pred)
        return np.array(preds)

def eval_k(k):
    knn_tmp = KNN(k=k)
    knn_tmp.fit(X_train, y_train)
    y_val_pred = knn_tmp.predict(X_val)
    acc_val = np.mean(y_val_pred == y_val)
    return float(acc_val)
candidates = [1,3,5,7,9,11,13,15]
scores = [(k, eval_k(k)) for k in candidates]
best_k, best_acc = max(scores, key=lambda x: x[1])
print("验证集最优 k:", best_k, "Acc:", f"{best_acc*100:.2f}%")

# ========== 3. 训练与测试 ==========
print("训练 KNN（其实只是存数据到Model，这里我们象征性地fit一下）...")
# 将原来的固定 k=10 改为使用验证集挑选出来的 best_k
knn = KNN(k=best_k)
knn.fit(X_train, y_train)
print("预测中...")
y_pred = knn.predict(X_test)

# ========== 4. 计算准确率 ==========
acc = np.mean(y_pred == y_test)
print(f"测试集最终准确率: {acc * 100:.2f}%")

# ========== 5. 可视化部分预测结果 ==========
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28).get(), cmap='gray')
    plt.title(f"预测:{y_pred[i]}, 真实:{y_test[i]}", fontdict={'family':'Microsoft YaHei', 'color':'black'})
    plt.axis('off')
plt.tight_layout()
plt.show()

# ===（保持不变）混淆矩阵可视化 ===
num_classes = int(np.max(y_test)) + 1
cm = np.zeros((num_classes, num_classes), dtype=np.int64)
for t, p in zip(y_test, y_pred):
    cm[int(t), int(p)] += 1

plt.figure(figsize=(6,5))
plt.imshow(cm if not hasattr(cm, "get") else cm.get(), interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.show()
```