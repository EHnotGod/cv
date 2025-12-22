import numpy as np
import json
import gzip
import time
import matplotlib.pyplot as plt

# --- matplotlib 中文支持 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        print("中文字体未找到，可能显示异常")

plt.rcParams['axes.unicode_minus'] = False
# ---------------------------


# --- 激活函数 ---

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# --- 两层 MLP 网络 ---

class TwoLayerMLP:

    def __init__(self, input_size, hidden_size, output_size, std=0.01):
        self.W1 = np.random.randn(hidden_size, input_size) * std
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * std
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)

        cache = {"A1": A1, "A2": A2}
        return A2, cache

    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        log_probs = np.log(A2 + 1e-9)
        cost = - (1 / m) * np.sum(Y * log_probs)
        return np.squeeze(cost)

    def backward(self, cache, X, Y):
        m = X.shape[1]
        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads, learning_rate):
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]

    def fit(self, X_train, Y_train, epochs, learning_rate, batch_size):
        m = X_train.shape[1]
        costs = []
        start_time = time.time()

        for i in range(epochs):
            perm = np.random.permutation(m)
            X_shuffled = X_train[:, perm]
            Y_shuffled = Y_train[:, perm]

            epoch_cost = 0.
            num_batches = m // batch_size

            for j in range(num_batches):
                begin = j * batch_size
                end = (j + 1) * batch_size

                X_batch = X_shuffled[:, begin:end]
                Y_batch = Y_shuffled[:, begin:end]

                A2, cache = self.forward(X_batch)
                cost = self.compute_loss(A2, Y_batch)
                epoch_cost += cost

                grads = self.backward(cache, X_batch, Y_batch)
                self.update_parameters(grads, learning_rate)

            if m % batch_size != 0:
                X_batch = X_shuffled[:, num_batches * batch_size:]
                Y_batch = Y_shuffled[:, num_batches * batch_size:]
                if X_batch.shape[1] > 0:
                    A2, cache = self.forward(X_batch)
                    cost = self.compute_loss(A2, Y_batch)
                    epoch_cost += cost
                    grads = self.backward(cache, X_batch, Y_batch)
                    self.update_parameters(grads, learning_rate)

            avg_cost = epoch_cost / (num_batches + (1 if m % batch_size else 0))
            costs.append(avg_cost)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"Epoch {i+1}/{epochs} - Loss: {avg_cost:.6f}")

        print(f"训练完成，用时 {time.time() - start_time:.2f}s")
        return costs

    def predict(self, X):
        A2, _ = self.forward(X)
        return np.argmax(A2, axis=0)

    def compute_accuracy(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == y_test) * 100


# --- MNIST 数据加载 ---

def load_mnist_data():
    with gzip.open('../../data/mnist.json.gz', 'rb') as f:
        train_set, _, test_set = json.load(f)

    X_train, y_train = train_set
    X_test, y_test = test_set

    index = 10000
    X_train, y_train = X_train[:index], y_train[:index]
    X_test, y_test = X_test[:index], y_test[:index]

    X_train = np.array(X_train).reshape(len(X_train), -1)
    X_test = np.array(X_test).reshape(len(X_test), -1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    Y_train = np.zeros((len(y_train), 10))
    Y_train[np.arange(len(y_train)), y_train] = 1

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T

    return X_train, Y_train, X_test, y_test


# --- 可视化 ---

def plot_loss(costs):
    plt.plot(costs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.show()


# --- 主程序 ---

def main():
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    LEARNING_RATE = 0.5
    EPOCHS = 100
    BATCH_SIZE = 128

    X_train, Y_train, X_test, y_test = load_mnist_data()

    mlp = TwoLayerMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    costs = mlp.fit(X_train, Y_train, EPOCHS, LEARNING_RATE, BATCH_SIZE)

    acc = mlp.compute_accuracy(X_test, y_test)
    print(f"测试集准确率: {acc:.2f}%")

    plot_loss(costs)


if __name__ == "__main__":
    main()
