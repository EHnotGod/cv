import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import nailorch
from nailorch import DataLoader
import nailorch.functions as F
from nailorch.datasets import MNIST
from model import MNISTNet
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 设置绘图风格和字体
plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
except:
    pass

def plot_learning_curves(history):
    """1. 绘制并保存学习曲线"""
    print("[1/6] 生成学习曲线 (learning_curves.png)...")
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(epochs, history['test_acc'], 'g-o', label='Test Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=100)
    plt.close()

def plot_confusion_matrix_and_report(y_true, y_pred):
    """2. 绘制混淆矩阵并保存文本分类报告"""
    print("[2/6] 生成混淆矩阵 (confusion_matrix.png)...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()

    print("[3/6] 生成分类报告 (test_report.txt)...")
    report = classification_report(y_true, y_pred, digits=4)
    
    # 额外：直接在控制台打印结果，防止用户以为没有输出
    print("\n" + "="*50)
    print("【测试集最终评估报告】")
    print("-" * 50)
    print(report)
    print("="*50 + "\n")
    
    # 计算每个类别的准确率用于后续绘图
    # 混淆矩阵对角线 / 该行总和
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    with open('test_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== MNIST Test Report ===\n\n")
        f.write(report)
        f.write("\n\n=== Per-Class Accuracy ===\n")
        for i, acc in enumerate(per_class_acc):
            f.write(f"Digit {i}: {acc:.2%}\n")
            
    return per_class_acc

def plot_per_class_accuracy(per_class_acc):
    """3. 绘制每个数字的准确率柱状图"""
    print("[4/6] 生成每类准确率图 (class_accuracy.png)...")
    digits = list(range(10))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(digits, per_class_acc, color='skyblue', edgecolor='navy')
    
    plt.title('Per-Class Accuracy (Which digit is hardest?)')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.xticks(digits)
    # 动态调整Y轴范围，让差异更明显
    min_acc = min(per_class_acc)
    plt.ylim(max(0.0, min_acc - 0.05), 1.0) 
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.savefig('class_accuracy.png', dpi=100)
    plt.close()

def visualize_filters(model):
    """4. 可视化第一层卷积核"""
    print("[5/6] 可视化卷积核 (conv1_filters.png)...")
    # 获取权重: shape (16, 1, 3, 3)
    W = model.conv1.W.data
    if nailorch.cuda.gpu_enable:
        W = nailorch.cuda.as_numpy(W)
    
    n_filters = W.shape[0]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('First Layer Convolution Filters (3x3 Weights)', fontsize=14)
    
    for i in range(n_filters):
        ax = axes[i//8, i%8]
        # 取出第i个卷积核，(1, 3, 3) -> (3, 3)
        w_img = W[i, 0] 
        # 归一化到 0-1 以便显示
        w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min() + 1e-8)
        
        ax.imshow(w_img, cmap='gray', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'Filter {i}', fontsize=8)
        
    plt.savefig('conv1_filters.png', dpi=100)
    plt.close()

def visualize_feature_maps(model, dataset):
    """5. 可视化特征图 (模型到底看到了什么)"""
    print("[6/6] 可视化特征图 (feature_maps.png)...")
    
    # 取第一张测试图片
    x, t = dataset[0] # x shape (1, 28, 28)
    
    # 准备输入
    if nailorch.cuda.gpu_enable:
        x_in = nailorch.cuda.as_cupy(x).reshape(1, 1, 28, 28)
    else:
        x_in = x.reshape(1, 1, 28, 28)
    
    # 手动运行第一层: Conv1 -> ReLU (不带池化，为了看清晰的特征)
    with nailorch.no_grad():
        layer_out = model.conv1(x_in)
        layer_out = F.relu(layer_out)
        
    # 获取数据并转回 numpy
    if nailorch.cuda.gpu_enable:
        feature_maps = nailorch.cuda.as_numpy(layer_out.data)
    else:
        feature_maps = layer_out.data
        
    # shape: (1, 16, 28, 28) -> (16, 28, 28)
    feature_maps = feature_maps[0]
    
    # 绘图
    fig = plt.figure(figsize=(12, 8))
    
    # 左边画原始图
    ax_orig = plt.subplot2grid((4, 5), (1, 0), rowspan=2, colspan=1)
    ax_orig.imshow(x.reshape(28, 28), cmap='gray')
    ax_orig.set_title(f'Input: {t}')
    ax_orig.axis('off')
    
    # 右边画 16 个特征图
    for i in range(16):
        # 计算网格位置 (0~3行, 1~4列)
        row = i // 4
        col = i % 4 + 1 
        ax = plt.subplot2grid((4, 5), (row, col))
        
        ax.imshow(feature_maps[i], cmap='viridis') # 使用 viridis 热力图显示激活程度
        ax.axis('off')
        ax.set_title(f'Map {i}', fontsize=8)
        
    plt.suptitle('Conv1 Feature Maps (What the network "sees")', fontsize=16)
    plt.savefig('feature_maps.png', dpi=100)
    plt.close()

def analyze_errors(model, dataset, y_true, y_pred, probs):
    """6. 错误分析 (单独保存为 error_analysis.png)"""
    print("      -> 生成错误分析图 (error_analysis.png)...")
    
    # 转换为 numpy 数组以便索引
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = np.array(probs)
    
    # 找到错误索引
    incorrect_mask = y_pred != y_true
    if np.sum(incorrect_mask) == 0:
        print("无错误样本！")
        return

    # 获取错误的置信度
    error_probs = probs[incorrect_mask]
    error_conf = np.max(error_probs, axis=1) # 预测类的置信度
    
    # 获取对应的原始索引
    error_indices = np.where(incorrect_mask)[0]
    
    # 组合成 (index, confidence) 列表并排序
    mistakes = []
    for i in range(len(error_indices)):
        mistakes.append({
            'idx': error_indices[i],
            'conf': error_conf[i],
            'pred': y_pred[error_indices[i]],
            'true': y_true[error_indices[i]]
        })
    
    # 按置信度降序排序
    mistakes.sort(key=lambda x: x['conf'], reverse=True)
    
    top_n = min(10, len(mistakes))
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(f'Top {top_n} "High Confidence" Errors', fontsize=14)
    
    for i in range(top_n):
        item = mistakes[i]
        idx = item['idx']
        
        # 从 dataset 获取原始图像
        img, _ = dataset[idx]
        img = img.reshape(28, 28)
        
        ax = axes[i//5, i%5]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"T:{item['true']} P:{item['pred']}\nConf:{item['conf']:.1%}", 
                     color='red', fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=100)
    plt.close()

def main():
    # 1. 准备模型和数据
    print(">>> 正在加载模型和数据...")
    test_set = MNIST(train=False)
    model = MNISTNet()
    
    if os.path.exists('mnist_net.npz'):
        model.load_weights('mnist_net.npz')
        print("模型权重加载成功。")
    else:
        print("警告: 未找到权重文件，正在使用随机初始化的模型！")
    
    if nailorch.cuda.gpu_enable:
        model.to_gpu()

    # 2. 绘制训练曲线 (如果存在)
    if os.path.exists('history.npy'):
        history = np.load('history.npy', allow_pickle=True).item()
        plot_learning_curves(history)
    else:
        print("跳过学习曲线 (未找到 history.npy)")

    # 3. 执行全量推理
    print(">>> 正在对测试集进行推理...")
    loader = DataLoader(test_set, batch_size=100, shuffle=False)
    all_preds = []
    all_labels = []
    all_probs = [] # 保存概率用于错误分析
    
    with nailorch.no_grad():
        for x, t in loader:
            if nailorch.cuda.gpu_enable:
                x_gpu = nailorch.cuda.as_cupy(x).reshape(-1, 1, 28, 28)
                y = model(x_gpu)
                prob = nailorch.functions.softmax(y).data
                prob = nailorch.cuda.as_numpy(prob)
            else:
                x_in = x.reshape(-1, 1, 28, 28)
                y = model(x_in)
                prob = nailorch.functions.softmax(y).data
            
            pred = np.argmax(prob, axis=1)
            
            all_preds.extend(pred)
            all_labels.extend(t)
            all_probs.extend(prob)

    # 4. 生成各类图表和报告
    
    # 显式计算并打印准确率
    final_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*40}")
    print(f"最终测试集准确率 (Overall Accuracy): {final_acc:.2%}")
    print(f"{'='*40}\n")
    
    # (a) 混淆矩阵 & 文本报告 (现在也会打印到控制台)
    per_class_acc = plot_confusion_matrix_and_report(all_labels, all_preds)
    
    # (b) 每类准确率柱状图
    plot_per_class_accuracy(per_class_acc)
    
    # (c) 卷积核可视化 (Weights)
    visualize_filters(model)
    
    # (d) 特征图可视化 (Activations)
    visualize_feature_maps(model, test_set)
    
    # (e) 错误分析
    analyze_errors(model, test_set, all_labels, all_preds, all_probs)
    
    print("\n>>> 所有分析已完成！请查看生成的 .png 图片和 test_report.txt 文件。")

if __name__ == '__main__':
    main()