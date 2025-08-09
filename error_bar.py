import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, precision_score, \
    recall_score
from model.dataset import CustomDataset
from model.network_class_ori import EnhancedVAE
from torch.utils.data import DataLoader
import random
import seaborn as sns
from Bio import SeqIO
import hashlib


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 创建噪声向量
def make_noise(batch_size, shape):
    tensor = torch.randn(batch_size, shape)
    noise = tensor.to(device, dtype=torch.float32)
    return noise


# 计算评估指标
def cal_auc(save_path, ymat, data):
    os.makedirs(save_path, exist_ok=True)
    ymat = ymat.flatten()
    data = data.flatten()

    # 计算AUROC
    fpr, tpr, _ = roc_curve(data, ymat)
    auroc = auc(fpr, tpr)

    # 计算AUPR
    precision, recall, _ = precision_recall_curve(data, ymat)
    aupr = auc(recall, precision)

    # 计算TSS
    y_pred = np.where(ymat >= 0.5, 1, 0)
    tn, fp, fn, tp = confusion_matrix(data, y_pred).ravel()
    tpr_value = tp / (tp + fn)
    fpr_value = fp / (fp + tn)
    tss = tpr_value - fpr_value

    # 计算其他指标
    f1 = f1_score(data, y_pred)
    precision_val = precision_score(data, y_pred)
    recall_val = recall_score(data, y_pred)

    # 保存指标
    results = pd.DataFrame({
        'AUROC': [auroc],
        'AUPR': [aupr],
        'TSS': [tss],
        'F1': [f1],
        'Precision': [precision_val],
        'Recall': [recall_val]
    })
    results.to_csv(os.path.join(save_path, 'evaluation_metrics.csv'), index=False)
    print('AUROC= %.4f | AUPR= %.4f | TSS= %.4f | precision= %.4f | recall= %.4f | f1_score= %.4f' % (
        auroc, aupr, tss, precision_val, recall_val, f1))
    return auroc, aupr, tss,recall_val

with_cuda = True

idx = 0
# 单模型预测函数
def predict_single_model(model_path, val_loader, input_dim_host):
    model = EnhancedVAE(464, input_dim_host, max_frequence=312524)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()

    eval_targets = []
    eval_outputs = []
    all_probs = []
    with torch.no_grad():
        for batchidx, (virus_data, host_data, target, eval_index, simi_data) in enumerate(val_loader):
            virus_data = virus_data.to(device)
            host_data = host_data.to(device)
            noise = make_noise(virus_data.shape[0], 10)

            # 模型预测
            output, _ = model(virus_data, host_data, noise)
            probs = torch.sigmoid(output).detach().cpu().numpy()
            all_probs.append(probs)
            eval_outputs.append(torch.sigmoid(output))
            eval_targets.append(target)

        if with_cuda:
            eval_outputs = torch.cat(eval_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            eval_targets = torch.cat(eval_targets, dim=0).detach().cpu().numpy()
        else:
            eval_outputs = torch.cat(eval_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
            eval_targets = torch.cat(eval_targets, dim=0).detach().numpy()

        auroc, aupr, tss, recall = cal_auc('./ensemble_results/idx', eval_outputs, eval_targets)
        # print(f"model {model_path} Performance: AUROC={auroc:.4f}, AUPR={aupr:.4f}, TSS={tss:.4f},recall={recall:.4f}")
    return np.concatenate(all_probs)


# 主函数
def main():
    # 设置全局变量
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # 加载数据
    coexistence_data = pd.read_csv("./data_VAE/SARS_association_vector.csv", index_col=0, header=None)
    virus_data = pd.read_csv("./data_2025_03_19_new_hamming/SARS_hamming_vector.csv", index_col=0, header=None)
    host_data = pd.read_csv("./data_VAE/SARS_hosts_vector.csv", index_col=0, header=None)

    # 数据预处理
    virus_data = torch.tensor(virus_data.values.astype(np.float32), dtype=torch.float32)
    host_data = torch.tensor(host_data.values.astype(np.float32), dtype=torch.float32)
    coexistence_data = torch.tensor(coexistence_data.values.astype(np.float32), dtype=torch.float32)

    # 创建数据集和数据加载器
    val_dataset = CustomDataset(virus_data, host_data, coexistence_data)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # ================== 关键修改部分 ==================
    # 1. 定义5个独立训练模型的路径
    model_paths = [
        "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_rs_2/model/pytorch_model499.pth",
        "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_rs_1/model/pytorch_model499.pth",
        # "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_rs_2/model/pytorch_model249.pth",
        # "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_rs_1/model/pytorch_model249.pth",
        # "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_rs_7/model/pytorch_model249.pth"
    ]

    # 2. 对每个模型进行预测并收集结果
    all_predictions = []
    for model_path in model_paths:
        print(f"Predicting with model: {model_path}")
        preds = predict_single_model(model_path, val_loader, host_data.shape[1])
        all_predictions.append(preds)

    # 3. 将预测结果转换为numpy数组 (5 models x n_samples)
    ensemble_preds = np.array(all_predictions)

    # 4. 计算每个样本的均值和标准差
    mean_preds = np.mean(ensemble_preds, axis=0)
    std_preds = np.std(ensemble_preds, axis=0)

    # 5. 保存集成预测结果
    os.makedirs("./ensemble_results", exist_ok=True)
    np.save("./ensemble_results/mean_predictions.npy", mean_preds)
    np.save("./ensemble_results/std_predictions.npy", std_preds)

    # 6. 计算评估指标
    targets = coexistence_data.numpy().flatten()
    auroc, aupr, tss = cal_auc("./ensemble_results/", mean_preds, targets)
    print(f"Ensemble Performance: AUROC={auroc:.4f}, AUPR={aupr:.4f}, TSS={tss:.4f}")

    # ================== 绘制误差棒图 ==================
    # 7. 随机选择50个样本进行可视化
    np.random.seed(42)
    sample_indices = np.random.choice(len(mean_preds), 50, replace=False)

    plt.figure(figsize=(15, 8))

    # 绘制均值点
    plt.errorbar(
        x=sample_indices,
        y=mean_preds[sample_indices],
        yerr=std_preds[sample_indices],
        fmt='o',
        color='blue',
        ecolor='lightgray',
        elinewidth=3,
        capsize=5,
        label='Mean Prediction ± 1 SD'
    )

    # 添加真实标签
    true_labels = targets[sample_indices]
    for i, idx in enumerate(sample_indices):
        color = 'green' if true_labels[i] == 1 else 'red'
        plt.plot(idx, true_labels[i], 's', markersize=10, color=color, alpha=0.7)

    # 图例和标签
    plt.title('Ensemble Predictions with Error Bars', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Prediction Probability', fontsize=14)
    plt.legend(['Positive Class', 'Negative Class', 'Prediction ± SD'], loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 保存和显示
    plt.tight_layout()
    plt.savefig("./ensemble_results/error_bars.png", dpi=300)
    plt.close()

    print("Error bar plot saved to ./ensemble_results/error_bars.png")

    # ================== 可选：绘制分布图 ==================
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=targets, y=mean_preds, inner="quartile", palette="muted")
    plt.title('Prediction Distribution by True Class')
    plt.xlabel('True Class')
    plt.ylabel('Prediction Probability')
    plt.savefig("./ensemble_results/class_distribution.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()