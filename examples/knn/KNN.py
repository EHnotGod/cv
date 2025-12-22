import matplotlib.pyplot as plt
import cupy as np
# 假如没有gpu的话，使用numpy
import json
import gzip
# ========== 1. 读取 MNIST ==========
datafile = '../../data/mnist.json.gz'
print('正在从 {} 中加载数据......'.format(datafile))
# 加载json数据文件
data = json.load(gzip.open(datafile))
print('Mnist 数据集加载完成')
# 读取到的数据区分训练集，验证集，测试集
train_set, val_set, test_set = data
X_train, y_train = train_set
X_test, y_test = test_set
# 为了演示快一点，先只用一小部分
index = 10000
X_train, y_train = X_train[:index], y_train[:index]
X_test, y_test = X_test[:index], y_test[:index]
# 展平并归一化
X_train = np.array(X_train).reshape(len(X_train), -1).astype(np.float32)
X_test = np.array(X_test).reshape(len(X_test), -1).astype(np.float32)
y_train = np.array(y_train); y_test = np.array(y_test)
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
# ========== 3. 训练与测试 ==========
print("训练 KNN（其实只是存数据到Model，这里我们象征性地fit一下）...")
knn = KNN(k=10)
knn.fit(X_train, y_train)
print("预测中...")
y_pred = knn.predict(X_test)
# ========== 4. 计算准确率 ==========
acc = np.mean(y_pred == y_test)
print(f"准确率: {acc * 100:.2f}%")
# ========== 5. 可视化部分预测结果 ==========
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28).get(), cmap='gray')
    plt.title(f"预测:{y_pred[i]}, 真实:{y_test[i]}", fontdict={'family':'Microsoft YaHei', 'color':'black'})
    plt.axis('off')
plt.tight_layout()
plt.show()