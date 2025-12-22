import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt
import matplotlib

# --- 设置 matplotlib 支持中文显示 ---
# SimHei 是一个常见的中文字体，您也可以替换为您系统上有的其他中文字体
# 例如 'Microsoft YaHei' (微软雅黑), 'FangSong' (仿宋) 等
try:
    # 尝试设置 SimHei 字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print("已设置字体为 SimHei。")
except:
    print("未找到 SimHei 字体，尝试使用 'Microsoft YaHei'...")
    try:
        # 如果 SimHei 失败，尝试 Microsoft YaHei
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        print("已设置字体为 Microsoft YaHei。")
    except:
        print("警告：未找到 'SimHei' 或 'Microsoft YaHei' 字体，中文显示可能异常。")

plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# ------------------------------------


# --- 1. 激活函数及其导数 ---
# 对应理论基础：第3.3节

def sigmoid(z):
    """Sigmoid 激活函数"""
    # 增加 np.clip 防止 z 值过大或过小导致 exp 溢出
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Sigmoid 激活函数的导数，输入为激活后的值 a"""
    # a = sigmoid(z)，导数 = a * (1 - a)
    return a * (1 - a)

def softmax(z):
    """Softmax 激活函数（用于输出层）"""
    # 为防止数值溢出，减去 z 中的最大值
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# --- 2. 神经网络类定义 ---

class TwoLayerMLP:
    """
    一个完全从零开始实现的两层多层感知机。
    结构: 输入层 -> 隐藏层 (Sigmoid) -> 输出层 (Softmax)
    """

    def __init__(self, input_size, hidden_size, output_size, std=0.01):
        """
        初始化网络参数（权重和偏置）。
        对应理论基础：第5.1节（参数W, b）

        参数:
        input_size (int): 输入层节点数 (MNIST 为 784)
        hidden_size (int): 隐藏层节点数 (例如 128)
        output_size (int): 输出层节点数 (MNIST 为 10)
        std (float): 权重初始化的标准差，一个较小的值有助于训练
        """
        # W1: (hidden_size, input_size), b1: (hidden_size, 1)
        self.W1 = np.random.randn(hidden_size, input_size) * std
        self.b1 = np.zeros((hidden_size, 1))

        # W2: (output_size, hidden_size), b2: (output_size, 1)
        self.W2 = np.random.randn(output_size, hidden_size) * std
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        """
        执行前向传播。
        对应理论基础：第5.1节

        参数:
        X (np.array): 输入数据，形状 (input_size, m)，m 为样本数

        返回:
        A2 (np.array): 输出层的激活值（预测概率），形状 (output_size, m)
        cache (dict): 存储中间变量，用于反向传播
        """
        # 第1层 (隐藏层)
        # 线性组合 Z1 = W1 * X + b1
        Z1 = np.dot(self.W1, X) + self.b1
        # 激活 A1 = sigmoid(Z1)
        A1 = sigmoid(Z1)

        # 第2层 (输出层)
        # 线性组合 Z2 = W2 * A1 + b2
        Z2 = np.dot(self.W2, A1) + self.b2
        # 激活 A2 = softmax(Z2)
        A2 = softmax(Z2)

        # 存储缓存，用于反向传播
        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }
        return A2, cache

    def compute_loss(self, A2, Y):
        """
        计算交叉熵损失。
        对应理论基础：第5.2节

        参数:
        A2 (np.array): 预测概率，形状 (output_size, m)
        Y (np.array): 真实标签 (one-hot)，形状 (output_size, m)

        返回:
        cost (float): 批量的平均损失
        """
        m = Y.shape[1] # 样本数量

        # 交叉熵损失: L = - (1/m) * sum(Y * log(A2))
        # 增加一个极小值 1e-9 防止 log(0)
        log_probs = np.log(A2 + 1e-9)
        cost = - (1 / m) * np.sum(Y * log_probs)

        cost = np.squeeze(cost) # 确保 cost 是一个标量
        return cost

    def backward(self, cache, X, Y):
        """
        执行反向传播，计算梯度。
        对应理论基础：第6节

        参数:
        cache (dict): 来自前向传播的缓存
        X (np.array): 输入数据，形状 (input_size, m)
        Y (np.array): 真实标签 (one-hot)，形状 (output_size, m)

        返回:
        grads (dict): 包含 W1, b1, W2, b2 梯度的字典
        """
        m = X.shape[1]

        # 从缓存中获取变量
        A1 = cache["A1"]
        A2 = cache["A2"]

        # --- 1. 计算输出层梯度 ---
        # 交叉熵损失 + Softmax 的导数是一个简洁的形式：dZ2 = A2 - Y
        # dZ2 形状: (output_size, m)
        dZ2 = A2 - Y

        # dW2 = (1/m) * dZ2 * A1.T
        # dW2 形状: (output_size, hidden_size)
        dW2 = (1 / m) * np.dot(dZ2, A1.T)

        # db2 = (1/m) * sum(dZ2)
        # db2 形状: (output_size, 1)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # --- 2. 计算隐藏层梯度 ---
        # dA1 = W2.T * dZ2
        # dA1 形状: (hidden_size, m)
        dA1 = np.dot(self.W2.T, dZ2)

        # dZ1 = dA1 * sigmoid'(A1)  (注意：这里用 A1 而不是 Z1)
        # dZ1 形状: (hidden_size, m)
        dZ1 = dA1 * sigmoid_derivative(A1)

        # dW1 = (1/m) * dZ1 * X.T
        # dW1 形状: (hidden_size, input_size)
        dW1 = (1 / m) * np.dot(dZ1, X.T)

        # db1 = (1/m) * sum(dZ1)
        # db1 形状: (hidden_size, 1)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }
        return grads

    def update_parameters(self, grads, learning_rate):
        """
        使用梯度下降更新参数。
        对应理论基础：第7.1节

        参数:
        grads (dict): 包含梯度的字典
        learning_rate (float): 学习率 alpha
        """
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]

    def fit(self, X_train, Y_train, epochs, learning_rate, batch_size):
        """
        训练神经网络模型。

        参数:
        X_train (np.array): 训练数据, (input_size, m_train)
        Y_train (np.array): 训练标签 (one-hot), (output_size, m_train)
        epochs (int): 迭代次数
        learning_rate (float): 学习率
        batch_size (int): 小批量大小
        """

        m = X_train.shape[1]
        costs = []

        print(f"开始训练...")
        print(f"总样本数: {m}, 批大小: {batch_size}, 学习率: {learning_rate}")

        start_time = time.time()

        for i in range(epochs):
            # --- 小批量梯度下降 (Mini-Batch) ---
            # 对应理论基础：第7.3节

            # 1. 打乱数据
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]

            epoch_cost = 0.
            num_batches = m // batch_size

            for j in range(num_batches):
                # 2. 提取一个小批量
                begin = j * batch_size
                end = (j + 1) * batch_size
                X_batch = X_shuffled[:, begin:end]
                Y_batch = Y_shuffled[:, begin:end]

                # 3. 前向传播
                A2, cache = self.forward(X_batch)

                # 4. 计算损失
                cost = self.compute_loss(A2, Y_batch)
                epoch_cost += cost

                # 5. 反向传播
                grads = self.backward(cache, X_batch, Y_batch)

                # 6. 更新参数
                self.update_parameters(grads, learning_rate)

            # 处理最后一个不满 batch_size 的批次
            if m % batch_size != 0:
                X_batch = X_shuffled[:, num_batches * batch_size:]
                Y_batch = Y_shuffled[:, num_batches * batch_size:]

                if X_batch.shape[1] > 0: # 确保不是空批次
                    A2, cache = self.forward(X_batch)
                    cost = self.compute_loss(A2, Y_batch)
                    epoch_cost += cost
                    grads = self.backward(cache, X_batch, Y_batch)
                    self.update_parameters(grads, learning_rate)

            avg_epoch_cost = epoch_cost / (num_batches + (1 if m % batch_size != 0 else 0))
            costs.append(avg_epoch_cost)

            # 打印训练进度
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Epoch {i+1}/{epochs} - 耗时: {time.time() - start_time:.2f}s - 平均损失: {avg_epoch_cost:.6f}")

        print(f"训练完成！总耗时: {time.time() - start_time:.2f}s")
        return costs

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        参数:
        X (np.array): 输入数据, (input_size, m)

        返回:
        predictions (np.array): 预测的类别 (0-9), 形状 (m,)
        """
        # 只需要前向传播
        A2, _ = self.forward(X)
        # A2 是 (10, m) 的概率矩阵，我们选择每列（每个样本）概率最大的索引
        predictions = np.argmax(A2, axis=0)
        return predictions

    def compute_accuracy(self, X_test, Y_test_labels):
        """
        计算模型在测试集上的准确率。

        参数:
        X_test (np.array): 测试数据, (input_size, m_test)
        Y_test_labels (np.array): 测试集原始标签 (非 one-hot), 形状 (m_test,)

        返回:
        accuracy (float): 准确率
        """
        predictions = self.predict(X_test)
        # Y_test_labels 必须是 (m_test,) 形状
        accuracy = np.mean(predictions == Y_test_labels) * 100
        return accuracy

# --- 3. 数据加载和预处理 ---

def load_mnist_data():
    """
    加载并预处理 MNIST 数据集。
    """
    print("正在加载 MNIST 数据集 (可能需要几分钟)...")
    # 1. 加载数据 (70000 个样本, 28x28=784 个特征)
    # as_frame=False 确保返回 numpy 数组
    # data_home 指定缓存目录
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, data_home='./mnist_cache', parser='liac-arff')
    print("数据加载完毕。")

    # 2. 归一化 (Normalization)
    # 对应理论基础：第9节（虽然我们没实现BN/LN，但输入归一化是必须的）
    # 将像素值从 [0, 255] 缩放到 [0, 1]
    X = X / 255.0

    # 3. 划分训练集和测试集 (MNIST 传统划分：60000 训练, 10000 测试)
    X_train, X_test = X[:60000], X[60000:]
    y_train_labels, y_test_labels = y[:60000], y[60000:]

    # 确保 y 是整数类型，以便 OneHotEncoder 使用
    y_train_labels = y_train_labels.astype(int)
    y_test_labels = y_test_labels.astype(int)

    # 4. One-Hot 编码
    # 将 (m,) 的标签 [7, 2, 1, ...] 转换为 (m, 10) 的矩阵
    # [[0,0,0,0,0,0,0,1,0,0],
    #  [0,0,1,0,0,0,0,0,0,0],
    #  [0,1,0,0,0,0,0,0,0,0], ...]
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    # Reshape(-1, 1) 是因为 encoder 需要 (m, 1) 的输入
    Y_train_onehot = encoder.fit_transform(y_train_labels.reshape(-1, 1))
    Y_test_onehot = encoder.transform(y_test_labels.reshape(-1, 1))

    # 5. 调整数据形状 (重要！)
    # 我们的网络需要 (features, m) 而不是 (m, features)
    # (60000, 784) -> (784, 60000)
    X_train_T = X_train.T
    X_test_T = X_test.T
    # (60000, 10) -> (10, 60000)
    Y_train_onehot_T = Y_train_onehot.T
    Y_test_onehot_T = Y_test_onehot.T

    print(f"X_train 形状: {X_train_T.shape}")
    print(f"Y_train (one-hot) 形状: {Y_train_onehot_T.shape}")
    print(f"X_test 形状: {X_test_T.shape}")
    print(f"Y_test (原始标签) 形状: {y_test_labels.shape}")

    return X_train_T, Y_train_onehot_T, X_test_T, y_test_labels, Y_test_onehot_T

# --- 4. 辅助函数：可视化 ---

def plot_loss(costs):
    """绘制训练过程中的损失下降曲线"""
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('平均损失 (Cost)')
    plt.xlabel('迭代次数 (Epochs)')
    plt.title('训练过程中的损失函数下降曲线')
    plt.grid(True)
    plt.show()

def show_predictions(X_test, y_test_labels, mlp_model, num_images=15):
    """随机抽取几张测试图片，显示预测结果"""

    # 从测试集中随机选择索引
    m_test = X_test.shape[1]
    indices = np.random.choice(m_test, num_images, replace=False)

    X_sample = X_test[:, indices]
    y_sample_labels = y_test_labels[indices]

    # 获取预测结果
    predictions = mlp_model.predict(X_sample)

    # 绘制图像
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(3, 5, i + 1)
        # 将 (784, 1) 的图像数据 reshape 回 (28, 28)
        img = X_sample[:, i].reshape(28, 28)
        plt.imshow(img, cmap='gray')

        # 预测正确显示绿色，错误显示红色
        color = 'green' if predictions[i] == y_sample_labels[i] else 'red'
        plt.title(f"真实: {y_sample_labels[i]}\n预测: {predictions[i]}", color=color)
        plt.axis('off')

    plt.suptitle("模型预测结果展示 (绿=正确, 红=错误)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 5. 主执行函数 ---

def main():
    # --- 超参数设置 ---
    INPUT_SIZE = 784     # 28x28 像素
    HIDDEN_SIZE = 128    # 隐藏层大小 (可以调整)
    OUTPUT_SIZE = 10     # 10 个类别 (0-9)
    LEARNING_RATE = 0.5  # 学习率 (可以调整)
    EPOCHS = 100          # 训练迭代次数 (可以调整)
    BATCH_SIZE = 128     # 小批量大小 (可以调整)

    # 1. 加载和准备数据
    X_train, Y_train, X_test, y_test_labels, Y_test = load_mnist_data()

    # 2. 初始化模型
    mlp = TwoLayerMLP(input_size=INPUT_SIZE,
                      hidden_size=HIDDEN_SIZE,
                      output_size=OUTPUT_SIZE)

    # 3. 训练模型
    costs = mlp.fit(X_train, Y_train,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    batch_size=BATCH_SIZE)

    # 4. 评估模型
    accuracy = mlp.compute_accuracy(X_test, y_test_labels)
    print(f"\n模型在 {X_test.shape[1]} 个测试样本上的准确率: {accuracy:.2f}%")

    # 5. 可视化
    plot_loss(costs)
    show_predictions(X_test, y_test_labels, mlp)

if __name__ == "__main__":
    main()