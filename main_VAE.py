import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model.trainer_VAE import Trainer
import os
from model.dataset import create_dataloaders
from Bio import SeqIO
from collections import Counter
import hashlib
import random


# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True  # 保证结果的确定性
#     torch.backends.cudnn.benchmark = False  # 禁用加速，确保可重复性
#
#
# # 设置随机种子
# seed = 5214394
# set_seed(seed)

import itertools

# 生成所有的三核苷酸组合
nucleotides = ['A', 'G', 'C', 'T']
triplets = [''.join(triplet) for triplet in itertools.product(nucleotides, repeat=3)]

# 初始化字典，值全为 0
triplet_dict = {triplet: 0 for triplet in triplets}

print(triplet_dict)


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


def start_trainer(virus_data, host_data, coexistence_data,virus_data_hamming,
                  target, input_dim_virus, input_dim_host, max_frequence, save_path):
    batch_size = 1024
    train_loader, test_loader, val_loader = create_dataloaders(
        virus_data, host_data, target,simi_data=virus_data_hamming, batch_size=batch_size, val_split=0.1, test_split=0.2
    )
    trainer = Trainer(save_path, train_dataloader=train_loader, test_dataloader=test_loader, val_dataloader=val_loader)
    trainer.create_model(input_dim_virus, input_dim_host, max_frequence)
    trainer._setup_model((coexistence_data == 0).sum().item() / (coexistence_data == 1).sum().item())
    trainer.model_runner(train_loader, test_loader, coexistence_data, target)


def main():
    max_frequence = -1
    # Initialize data_2025_07_22
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # virus_data = pd.read_csv("./data_VAE/viruses_vector.csv", index_col=0)
    virus_data_ = []
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
    #
    # host_data = pd.read_csv("./data_VAE/hosts_vector.csv", index_col=0, header=None)
    # # coexistence_data = pd.read_csv("./data_VAE/association_simulated_vector.csv", index_col=0,header=None)
    coexistence_data = pd.read_csv("./data_VAE/association_vector.csv", index_col=0, header=None)
    # virus_data = pd.read_csv("./data_VAE/viruses_vector.csv", index_col=0, header=None)
    # coexistence_data = pd.read_csv("./data_repeat/association_fold_vector.csv", index_col=0, header=None)
    # host_data = pd.read_csv("./data_repeat/hosts_fold_vector.csv", index_col=0, header=None)

    # host_data = pd.read_csv("./data_even/hosts_vector(1).csv", index_col=0, header=None)
    # coexistence_data = pd.read_csv("./data_even/association_vector(1).csv", index_col=0, header=None)
    # virus_data = pd.read_csv("./data_repeat/viruses_fold_vector.csv", index_col=0, header=None)
    # virus_data = np.array(virus_data, dtype=np.float32)

    # for i in range(virus_data.shape[0]):
    #     data_2025_07_22 = np.array(virus_data[i])
    #     if max_frequence < data_2025_07_22.max():
    #         max_frequence = data_2025_07_22.max()

    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding512_sampler_trans_42_vec_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_repeat_data_lstm_pool_50/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_fixed_sas_host_lstm_pool_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_fixed_sas_ori_100/"
    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_fixed_sas_repeat_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_fixed_sas_ori_nosample_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_fixed_sas_ori_nosample_pool_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_no_sampler_trans_decoder_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_sampler_no_pos_trans_decoder_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192batch_3ker_embedding1024_pos_trans_decoder_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_pos_virus_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_pos_concathamming_en_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_hanmmingvirusdata_concat_conn_decoder_no_mmp_100_/"
    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_virusdata_nodecoder_samplerpos_512_300/"
    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding8192_hammingdata_nodecoder_samplerpos_138_300/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_virusdata_concat_conn_decoder_mmp_100_/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_hammingdata_ori_100_normal/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024batch_3ker_embedding1024_hammingdata_ori_100_seedori_repeat_1/"
    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_8192_repeat_sampler_posres_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_concat_nosample_posres_learn_ran/"
    # save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_sample_posres_learn_517/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_concat_nosample_posres_ran_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_concat_ran_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_posres_nolearn_concat_rantime_100/"
    save_path = "./res_VAE/VAE_sequence_200epoch_noise_pos_1024_repeat_de_nosample_noposres_nolearn_concat_rantime_200/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_decoder_batchno_cnn_cross_VAE_KL_200/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_cnn_cls_x`decoder_GBLoss_KL_100/"
    save_path = "./res_VAE/VAE_sequence_3ker_noise_pos_1024_concat_norm_noise_z_cnn_cls_x`decoder_nonormdata_KL_100/"
    save_path = "./res_VAE/VAE_hanmming_noise_pos_1024_normmlp_cnn_cls_x_decoder_nosigmoid_100/"
    save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_deep/"
    save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_64/"
    save_path = "./res_VAE/VAE_new_hamming_data_mlpboth_cnn_512_w_new_20250722_5/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # virus_data = torch.tensor(np.array(virus_data), dtype=torch.float32)
    # virus_data = torch.tensor(np.array(virus_data), dtype=torch.long)

    # host_data = np.array(host_data, dtype=np.float32)
    # host_data = torch.tensor(host_data, dtype=torch.float32)
    # # host_data = torch.tensor(host_data, dtype=torch.long)
    #
    coexistence_data = torch.tensor(np.array(coexistence_data, dtype=np.float32), dtype=torch.float32)
    # host_data = 1 - (host_data - torch.min(host_data)) / (torch.max(host_data) - torch.min(host_data))
    # # virus_data = 1 - (virus_data - torch.min(virus_data)) / (torch.max(virus_data) - torch.min(virus_data))
    # to_valid_data_num = 0
    # input_dim_virus = virus_data.shape[1] - to_valid_data_num
    # input_dim_host = host_data.shape[1]

    import seaborn as sns
    import matplotlib.pyplot as plt
    # sns.heatmap(coexistence_data, cmap="viridis")
    # # 显示图像
    # plt.savefig('coexistence_data.png', dpi=400)
    # plt.close()
    # target = coexistence_data  # Train model
    # max_frequence = int(max_frequence + 1)

    virus_data_hamming = pd.read_csv("./data_VAE/viruses_hamming_vector.csv", index_col=0, header=None)
    virus_data_hamming = np.array(virus_data_hamming)
    virus_data_hamming = torch.tensor(virus_data_hamming,dtype=torch.int64)

    # plt.figure(figsize=(10, 6))
    #
    # sns.kdeplot(virus_data_hamming, fill=True, color="blue", alpha=0.5, legend=False)
    # plt.title("Density Plot of Virus Hamming Distances", fontsize=14)
    # plt.xlabel("Hamming Distance", fontsize=12)
    # plt.ylabel("Density", fontsize=12)
    # plt.show()

    # virus_data_hamming = torch.tensor(virus_data_hamming,dtype=torch.float32)
    virus_data_hamming = 1 - (virus_data_hamming - torch.min(virus_data_hamming)) / (torch.max(virus_data_hamming) - torch.min(virus_data_hamming))
    # plt.figure(figsize=(10, 6))
    # virus_data_hamming_f = virus_data_hamming.flatten()
    # sns.kdeplot(virus_data_hamming_f, fill=True, color="blue", alpha=0.5, legend=False)
    # plt.title("Density Plot of Virus Hamming Distances", fontsize=14)
    # plt.xlabel("Hamming Distance whole norm", fontsize=12)
    # plt.ylabel("Density", fontsize=12)
    # plt.show()

    max_frequence_hamming = virus_data_hamming.max()+1
    max_frequence = max_frequence if max_frequence > max_frequence_hamming else max_frequence_hamming
    # max_frequence_host = int(host_data.max()+1)
    # max_frequence = [max_frequence,max_frequence_host]
    print(max_frequence)
    # start_trainer(virus_data, host_data, coexistence_data,virus_data_hamming, target, input_dim_virus, input_dim_host, max_frequence,
    #               save_path)
    start_trainer(1, 1, coexistence_data,virus_data_hamming, 1, 464, 644, max_frequence,
                  save_path)

if __name__ == "__main__":
    main()
