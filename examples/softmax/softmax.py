import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import platform

# --- 1. 中文字体设置 ---
def set_chinese_font():
    """设置 matplotlib 支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    system = platform.system()
    font_names = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']
    if system == 'Windows':
        font_names.extend(['KaiTi', 'FangSong'])
    elif system == 'Darwin':  # macOS
        font_names.extend(['Arial Unicode MS'])
    
    for font in font_names:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"中文字体设置为: {font}")
            break
        except Exception:
            continue

# 在脚本开始时设置字体
set_chinese_font()

# --- 2. 数据加载与预处理 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

X_train_full = train_data.data.numpy().reshape(-1, 28*28).astype("float32") / 255.0
y_train_full = train_data.targets.numpy()
X_test       = test_data.data.numpy().reshape(-1, 28*28).astype("float32") / 255.0
y_test       = test_data.targets.numpy()

X_val, y_val   = X_train_full[50000:], y_train_full[50000:]
X_train, y_train = X_train_full[:50000], y_train_full[:50000]

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# --- 3. Softmax 类 ---
class NumpySoftmax:
    def __init__(self, input_dim, num_classes, lr=0.1, reg=1e-4):
        self.W = 0.001 * np.random.randn(input_dim, num_classes)
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.reg = reg

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def compute_loss_and_grads(self, X, y):
        num_samples = X.shape[0]
        scores = X.dot(self.W) + self.b
        probs = self.softmax(scores)
        loss = -np.mean(np.log(probs[np.arange(num_samples), y])) + 0.5 * self.reg * np.sum(self.W**2)
        dscores = probs
        dscores[np.arange(num_samples), y] -= 1
        dscores /= num_samples
        dW = X.T.dot(dscores) + self.reg * self.W
        db = np.sum(dscores, axis=0, keepdims=True)
        return loss, dW, db

    def train(self, X, y, X_val, y_val, epochs=100, batch_size=256):
        n = X.shape[0]
        history = {'loss': [], 'val_accuracy': []}
        
        print("--- 开始训练 ---")
        for epoch in range(epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            epoch_losses = []
            
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                loss, dW, db = self.compute_loss_and_grads(X_batch, y_batch)
                self.W -= self.lr * dW
                self.b -= self.lr * db
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            val_acc = self.accuracy(X_val, y_val)
            history['val_accuracy'].append(val_acc)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"轮次 {epoch:3d}/{epochs}, 损失={avg_loss:.4f}, 验证集准确率={val_acc:.4f}")
        
        print("--- 训练完成 ---")
        return history

    def predict(self, X):
        scores = X.dot(self.W) + self.b
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)

# --- 4. 绘图函数（保存为图片） ---
def plot_history(history, save_path='training_history.png'):
    """绘制训练损失与验证准确率，并保存到文件"""
    plt.figure(figsize=(12, 5))
    # 损失
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('训练损失曲线')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    # 验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], color='orange')
    plt.title('验证集准确率曲线')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (Accuracy)')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练历史图已保存至: {save_path}")
    return save_path

def plot_weights(W, save_path='weights.png'):
    """可视化 Softmax 的权重，并保存到文件"""
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        weight_img = W[:, i].reshape(28, 28)
        weight_img = (weight_img - np.min(weight_img)) / (np.max(weight_img) - np.min(weight_img) + 1e-12)
        plt.imshow(weight_img, cmap='viridis')
        plt.title(f'数字 {i} 的模板')
        plt.axis('off')
    plt.suptitle('模型学习到的权重 (模板)', fontsize=16)
    plt.savefig(save_path)
    plt.close()
    print(f"权重图已保存至: {save_path}")
    return save_path

def plot_confusion_matrix_heatmap(y_true, y_pred, save_path='confusion_matrix.png'):
    """绘制混淆矩阵热图，并保存到文件"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('混淆矩阵热图')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵图已保存至: {save_path}")
    return save_path

def plot_misclassified_examples(X, y_true, y_pred, n=10, save_path='misclassified.png'):
    """显示预测错误的样本，并保存到文件"""
    misclassified_idx = np.where(y_true != y_pred)[0]
    if len(misclassified_idx) == 0:
        print("恭喜！没有预测错误的样本。")
        return None
        
    n = min(n, len(misclassified_idx))
    sample_idx = np.random.choice(misclassified_idx, n, replace=False)
    
    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(sample_idx):
        cols = (n + 1) // 2
        plt.subplot(2, cols, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray_r')
        plt.title(f"真实: {y_true[idx]}\n预测: {y_pred[idx]}", color='red')
        plt.axis('off')
    plt.suptitle('错误分类样本示例', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"错误样本图已保存至: {save_path}")
    return save_path

# --- 5. 主执行流程 ---
if __name__ == "__main__":
    # 1) 初始化与训练
    clf = NumpySoftmax(input_dim=28*28, num_classes=10, lr=0.1, reg=1e-4)
    history = clf.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=256)

    # 2) 测试集评估
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print("\n--- 最终测试评估 ---")
    print(f"测试集准确率: {acc:.4f}")

    # 3) 生成图表
    print("\n--- 正在生成图表 ---")
    plot_history(history, 'training_history.png')
    plot_confusion_matrix_heatmap(y_test, y_pred, 'confusion_matrix.png')
    plot_weights(clf.W, 'weights.png')
    plot_misclassified_examples(X_test, y_test, y_pred, n=10, save_path='misclassified.png')

    print("\n所有图表已保存：training_history.png, confusion_matrix.png, weights.png, misclassified.png")
