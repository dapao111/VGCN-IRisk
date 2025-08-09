import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from model.dataset import create_dataloaders
from Bio import SeqIO
from collections import Counter
import hashlib
from model.dataset import CustomDataset
# from model.network_VAE import VAE_transmission_model
# from model.network_VAE_decoderconc import VAE_transmission_model
from model.network_class_ori import VAE_transmission_model, EnhancedVAE
from torch.optim.swa_utils import AveragedModel, SWALR

from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.autograd import Variable, grad
from sklearn.metrics import roc_curve, auc, accuracy_score, r2_score, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
# from torchsummary import summary
import random

# import pycuda.autoprimaryctx
device = torch.device("cuda")
with_cuda = True
import time

# seed = int(time.time() / 333) + int(time.time() / 777) ^ 2
#
seed = 42


# seed =7466175
# seed = 5210272
# seed = 5214394

# seed = 7450601
# seed = 5213862
# seed = 5210278
# seed = 7451338


# seed =7451407
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保证结果的确定性
    torch.backends.cudnn.benchmark = False  # 禁用加速，确保可重复性


def make_noise(batch_size, shape, volatile=False):
    tensor = torch.randn(batch_size, shape)
    noise = Variable(tensor, volatile)
    noise = noise.to(device, dtype=torch.float32)
    return noise


def eval_result_by_extern(df_path):
    unlikely_hosts = []
    with open('./sas_predict_results/unlikely.txt') as f:
        for line in f.readlines():
            unlikely_hosts.append(line.split('\n')[0])

    likely_hosts = []
    with open('./sas_predict_results/likely.txt') as f:
        for line in f.readlines():
            likely_hosts.append(line.split('\n')[0])

    receptor_hosts = []
    with open('./sas_predict_results/receptor.txt') as f:
        for line in f.readlines():
            receptor_hosts.append(line.split('\n')[0])

    observed_hosts = []
    with open('./sas_predict_results/observed.txt') as f:
        for line in f.readlines():
            observed_hosts.append(line.split('\n')[0])

    df = pd.read_csv(df_path)
    unlikely_df = df[df['V-H'].isin(unlikely_hosts)]
    print('unlikely: ' + str(sum(unlikely_df['Pro'].to_list()) / len(unlikely_df)))

    label_0_df = df[df['Label'] == 0]
    print('label 0: ' + str(sum(label_0_df['Pro'].to_list()) / len(label_0_df)))

    label_1_df = df[df['Label'] == 1]
    print('label 1: ' + str(sum(label_1_df['Pro'].to_list()) / len(label_1_df)))

    likely_df = df[df['V-H'].isin(likely_hosts)]
    print('likely: ' + str(sum(likely_df['Pro'].to_list()) / len(likely_df)))

    receptor_df = df[df['V-H'].isin(receptor_hosts)]
    print('receptor: ' + str(sum(receptor_df['Pro'].to_list()) / len(receptor_df)))

    observed_df = df[df['V-H'].isin(observed_hosts)]
    print('observed: ' + str(sum(observed_df['Pro'].to_list()) / len(observed_df)))


def cal_auc(save_path, ymat, data):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    y_mat_df = pd.DataFrame(ymat)
    y_mat_df.to_csv(save_path + "yamt_eval_sas_result.csv", mode='w', index=False, header=False)
    y_label_df = pd.DataFrame(data)
    y_label_df.to_csv(save_path + "ylabel_sas_result.csv", mode='w', index=False, header=False)
    ymat = ymat.flatten()
    data = data.flatten()

    # 计算 AUROC 和 AUPR
    fpr, tpr, rocth = roc_curve(data, ymat)
    auroc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 画对角线，表示随机猜测
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path + 'roc.pdf', format='pdf')
    plt.close()
    precision, recall, prth = precision_recall_curve(data, ymat)
    aupr = auc(recall, precision)

    # 预测类别
    y_pred = np.where(np.array(ymat) >= 0.5, 1, 0)

    # 计算 TSS
    tn, fp, fn, tp = confusion_matrix(data, y_pred).ravel()
    tpr_value = tp / (tp + fn)
    fpr_value = tn / (fp + tn)
    tss = tpr_value + fpr_value - 1

    # 计算其他指标
    f1_s = f1_score(data, y_pred, average='macro')
    f1_scores = 2 * (precision * recall) / (precision + recall)
    precisions = precision.mean()
    recall_1 = recall.mean()
    f1_score_ = f1_scores.mean()
    precision = precision_score(data, y_pred)
    recall = recall_score(data, y_pred)
    # 将结果记录到 CSV 文件
    thresholds = np.arange(0, 1.01, 0.02)  # 从0到1的阈值
    recall_values = []
    fpr_values = []
    precisions = []
    for threshold in thresholds:
        y_pred = np.where(ymat >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(data, y_pred).ravel()
        recall_values.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_values.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    plt.figure(figsize=(30, 15))
    plt.plot(thresholds, recall_values, label='Recall', color='blue')
    plt.plot(thresholds, fpr_values, label='FPR', color='red')
    plt.plot(thresholds, precisions, label='Precision', color='pink')
    plt.xticks(np.arange(0, 1.01, 0.02))  # 设置 x 轴刻度为 0 到 1，步长为 0.01
    plt.title('Recall and FPR vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    # plt.savefig('./ymat_hans_recall-fpr-precision_xticks.pdf', format='pdf')
    plt.savefig(save_path + 'fpr-precision_xticks_eval_sas.pdf', format='pdf')
    plt.close()
    results = pd.DataFrame({
        'AUROC': [auroc],
        'AUPR': [aupr],
        'TSS': [tss],  # 添加 TSS
        'precision_mean': [precisions],
        'recall_mean': [recall_1],
        'f1_score_mean': [f1_score_],
        'precision_0.5': [precision],
        'recall_0.5': [recall],
        'f1_score_0.5': [f1_s],
        'seed': seed
    })

    # 将结果保存到 CSV 文件
    results.to_csv(save_path + 'fold_result.csv', mode='a', header=not os.path.exists(save_path + 'fold_result.csv'),
                   index=False)
    print('eval res:\n')

    print('AUROC= %.4f | AUPR= %.4f | TSS= %.4f | precision= %.4f | recall= %.4f | f1_score= %.4f' % (
        auroc, aupr, tss, precision, recall, f1_s))

    return auroc, aupr, tss, recall


def start_predict(virus_data, host_data, coexistence_data,
                  target, input_dim_virus, input_dim_host, max_frequence, save_path):
    batch_size = 512
    val_dataset = CustomDataset(virus_data, host_data, target)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model = VAE_transmission_model(470, 644, 128, max_frequence, 1024, 0.5)
    model = EnhancedVAE(464, 644, max_frequence)

    model = nn.DataParallel(model)

    # state_dict = torch.load(save_path + '/model/pytorch_model899.bin')
    # for name, param in state_dict.items():
    #     print(name)
    # loaded_dict = torch.load(save_path + '/model/pytorch_model149.pth')
    # model.state_dict = loaded_dict
    #
    model.load_state_dict(torch.load(save_path + '/model/pytorch_model849.pth'), strict=True)

    # model.load_state_dict(torch.load(save_path + '/model/pytorch_model99.bin'),strict=True)

    # swa_model = AveragedModel(model)
    # model = swa_model
    # model.load_state_dict(torch.load(save_path + '/model/last_swa_model.bin'))

    model.to(device)
    # summary(model, input_size=((8192, 644, 64),(8192,464,644),(8192,10)))  # 根据你的输入调整形状
    model.eval()
    eval_outputs = []
    #middle_results_record
    eval_outputs_logit_list = []
    combined_z_list = []
    host_features_list = []
    virus_features_list = []
    extracted_decoded_output_list = []
    decoded_output_list = []
    eval_targets = []
    eval_loss = 0
    eval_indices = []

    with torch.no_grad():
        for batchidx, (virus_data, host_data, target, eval_index, simi_data) in enumerate(val_loader):
            if torch.all(simi_data != 0):
                virus_data, host_data, target, simi_data = virus_data.to(device), host_data.to(
                    device), target.to(
                    device), simi_data.to(device)
            else:
                virus_data, host_data, target = virus_data.to(device), host_data.to(device), target.to(
                    device)
            print(virus_data.shape[0])
            noise = make_noise(virus_data.shape[0], 10)
            # Forward pass
            # output, decoded_output, host_features, virus_features_norm, extracted_decoded_output, combined_z = model(
            #     virus_data, host_data, noise)
            output, decoded_output = model(virus_data, host_data, noise)
            # Collect outputs for AUC calculation
            eval_outputs.append(torch.sigmoid(output))
            # eval_outputs.append(torch.sigmoid(output[:, 1]))
            # eval_outputs.append(output[:, 1])

            ##record middle results
            eval_outputs_logit_list.append(output)
            # combined_z_list.append(combined_z)
            # host_features_list.append(host_features)
            # virus_features_list.append(virus_features_norm)
            # extracted_decoded_output_list.append(extracted_decoded_output)
            # decoded_output_list.append(decoded_output[1])

            eval_targets.append(target)
            eval_indices.append(eval_index)

    if with_cuda:
        eval_outputs = torch.cat(eval_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
        eval_targets = torch.cat(eval_targets, dim=0).detach().cpu().numpy()
        print(eval_outputs.shape,eval_targets.shape)
        # combined_z_all = torch.cat(combined_z_list, dim=0).detach().cpu().numpy()
        # host_features_all = torch.cat(host_features_list, dim=0).detach().cpu().numpy()
        # virus_features_all = torch.cat(virus_features_list, dim=0).detach().cpu().numpy()
        eval_outputs_logit = torch.cat(eval_outputs_logit_list, dim=0).detach().cpu().numpy()
        # extracted_decoded_output = torch.cat(extracted_decoded_output_list, dim=0).detach().cpu().numpy()
        # decoded_output_all = torch.cat(decoded_output_list, dim=0).detach().cpu().numpy()
    else:
        eval_outputs = torch.cat(eval_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
        eval_targets = torch.cat(eval_targets, dim=0).detach().numpy()

    eval_outputs_01 = (eval_outputs_logit - eval_outputs_logit.min()) / (
                eval_outputs_logit.max() - eval_outputs_logit.min())

    coexistence_data_sas_index = pd.read_csv("./data_VAE/SARS_association_vector.csv", index_col=0, header=None)
    eval_indices_last = np.concatenate(eval_indices)
    data_index = coexistence_data_sas_index.index.tolist()
    coexistence_data_sas_index = np.array(coexistence_data_sas_index)
    sas_labels = [coexistence_data_sas_index[i] for i in eval_indices_last]
    # 创建 DataFrame 并保存为 CSV
    eval_sas_labels_df = pd.DataFrame(
        np.array(sas_labels),
        index=[data_index[i] for i in eval_indices_last],  # 使用 coexistence_data.index 对应的索引
        columns=["Label"]
    )
    auroc, aupr, tss, recall = cal_auc('./20250722_sas_predict_results/new_data_mlpboth/', eval_outputs, eval_targets)

    eval_sas_labels_df.to_csv('./20250722_sas_predict_results/new_data_mlpboth/' + "/sas_eval_idx_labels.csv",
                              index=True,
                              header=True)
    eval_sas_labels_df["Pro"] = np.array(eval_outputs).flatten()  # 处理多维情况
    eval_sas_labels_df.index.name = "V-H"
    eval_sas_labels_df.to_csv(
        './20250722_sas_predict_results/new_data_mlpboth/sas_predict_results.csv',
        index=True,
        header=True
    )

    eval_result_by_extern('./20250722_sas_predict_results/new_data_mlpboth/sas_predict_results.csv')

    # 指定保存目录
    save_dir = './20250722_sas_predict_results/new_data_mlpboth/'
    os.makedirs(save_dir, exist_ok=True)

    # 保存所有变量
    np.savetxt(os.path.join(save_dir, 'logits.csv'), eval_outputs_logit, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'sigmoid_logits.csv'), eval_outputs, delimiter=',')
    # np.savetxt(os.path.join(save_dir, 'combined_z.csv'), combined_z_all, delimiter=',')
    # np.savetxt(os.path.join(save_dir, 'decoded_output.csv'), decoded_output_all, delimiter=',')
    # np.savetxt(os.path.join(save_dir, 'host_features.csv'), host_features_all, delimiter=',')
    # np.savetxt(os.path.join(save_dir, 'virus_features_norm.csv'), virus_features_all, delimiter=',')
    # np.savetxt(os.path.join(save_dir, 'extracted_decoded_output.csv'), extracted_decoded_output, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'normalized_01_logits.csv'), eval_outputs_01, delimiter=',')

    return auroc, aupr, tss, recall


def kmer_hash_vector(seq, k=3, vec_length=128):
    vec = np.zeros(vec_length)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        kmer_str = str(kmer)
        hash_val = int(hashlib.sha256(kmer_str.encode('utf-8')).hexdigest(), 16) % vec_length
        # vec[hash_val] += 1  # Record frequency
        # if kmer in triplet_dict:
        #     triplet_dict[kmer] += 1
        # hash_val = hash(kmer) % vec_length
        vec[hash_val] += 1  # 记录频率
        # triplet_vector = np.array(list(triplet_dict.values()))
    return vec


def main():
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    max_frequence = -1

    # virus_data_ = []
    # for record in SeqIO.parse("./data_VAE/viruses.fasta", 'fasta'):
    #     seq = []
    #     for seq_ in record.seq:
    #         if seq_ in char_to_idx:
    #             seq.append(char_to_idx[seq_])
    #     vector = kmer_hash_vector(record.seq, k=3, vec_length=64)
    #     if max_frequence < vector.max():
    #         max_frequence = vector.max()
    #
    #     virus_data_.append(np.array(vector))
    #
    # virus_data = np.repeat(virus_data_, repeats=644, axis=0)

    # virus_data = pd.read_csv("./data_VAE/SARS_hamming_vector.csv", index_col=0, header=None)
    coexistence_data = pd.read_csv("./data_VAE/SARS_association_vector.csv", index_col=0, header=None)
    virus_data = pd.read_csv("./data_2025_03_19_new_hamming/SARS_hamming_vector.csv", index_col=0, header=None)

    host_data = pd.read_csv("./data_VAE/SARS_hosts_vector.csv", index_col=0, header=None)
    virus_data = np.array(virus_data, dtype=np.float32)

    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding_shuffle_late_100/"

    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding_wuhui_even_sha256/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding_AEloss/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_2048batch_3ker_embedding_AEloss_300/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding_sampler_42_tss/"
    save_path = "./VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_sampler_test/"
    save_path = "VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_hanmmingvirusdata_concat_conn_decoder_1024_100"
    # save_path = "VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_sampler_test_decoder"
    # save_path = "VAE_sequence_200epoch_noise_pos_1024_repeat_de_concat_posres"
    save_path = "VAE_sequence_200epoch_noise_pos_1024_repeat_de_concat_nosample_posres_learn_ran"
    # save_path = "VAE_sequence_200epoch_noise_pos_1024_repeat_de_concat_nosample_posres_ran_100"
    # save_path = "VAE_sequence_200epoch_noise_pos_8192_repeat_de_nosample_posres_ran_100"
    # save_path ="VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_virus_data_meanconcat_ran_100"
    save_path = "VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_concat_trans_ran_100"
    save_path = "VAE_sequence_200epoch_noise_pos_8192_repeat_de_nosample_posres_nolearn_concat_trans_ran_100"
    save_path = "VAE_temperoary_model"

    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_concat_ran_s300"
    # save_path ="./res_VAE/VAE_sequence_200epoch_noise_pos_8192_repeat_de_nosample_posres_nolearn_concat_ran_500"

    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_concat_ran_500/"
    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_concat_ran442_500"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_noposres_nolearn_concat_rantime_100"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_di_200"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_di_cross_200/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_di_host_embedding_cross_200"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_di_concat_batchno_cross_200"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_concat_batchno_cnn_cross_200"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_concat_norm_decoder_batchno_cnn_cross_200/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_repeat_concat_norm_decoder_batchno_cnn_cross_200/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_non_pos_1024_concat_norm_noise_z_decoder_batchno_cnn_cross_200"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_decoder_batchno_cnn_cross_VAE_KL_200/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_decoder_batchno_cnn_cross_VAE_KL_200/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_cnn_cls_decoder_GBLoss_KL_100/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_cnn_cls_x`decoder_nonormdata_KL_100/"
    save_path = "./res_VAE/VAE_hanmming_noise_pos_1024_normmlp_cnn_cls_x_decoder_nosigmoid_100/"

    save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_2000/"
    # save_path = "./"
    # save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_deep/"
    save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_deep/"
    # save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_64"
    save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_w_new_20250722_5"
    # save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_w_new_20250722_4"
    # save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_1/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # virus_data = torch.tensor(np.array(virus_data), dtype=torch.long)
    virus_data = torch.tensor(np.array(virus_data), dtype=torch.float32)
    host_data = np.array(host_data, dtype=np.float32)
    host_data = torch.tensor(host_data, dtype=torch.float32)

    coexistence_data = torch.tensor(np.array(coexistence_data, dtype=np.float32), dtype=torch.float32)
    host_data = 1 - (host_data - torch.min(host_data)) / (torch.max(host_data) - torch.min(host_data))
    # virus_data = 1 - (virus_data - torch.min(virus_data)) / (torch.max(virus_data) - torch.min(virus_data))
    to_valid_data_num = 0
    input_dim_virus = virus_data.shape[1] - to_valid_data_num
    input_dim_host = host_data.shape[1]

    import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.heatmap(coexistence_data, cmap="viridis")
    # 显示图像
    # plt.savefig('coexistence_data.png', dpi=400)
    # plt.close()
    target = coexistence_data  # Train model
    # max_frequence = int(22669 + 1)
    max_frequence = 312524

    auroc, aupr, tss, recall = start_predict(virus_data, host_data, coexistence_data, target, input_dim_virus,
                                             input_dim_host, max_frequence,
                                             save_path)

    return auroc, aupr, tss, recall


if __name__ == "__main__":
    # recall_high = -1
    # tss_high = -1
    # record_seed = 0
    # for i in range(0,10000):
    #     set_seed(i)
    #     auroc, aupr, tss, recall = main()
    #     if  tss>tss_high:
    #          record_seed = i
    #          recall_high = recall
    #          tss_high = tss
    #
    #     print(recall_high,tss_high,record_seed)
    set_seed(seed)
    main()
