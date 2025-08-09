import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import numpy as np
from torch.utils.data import ConcatDataset

from sklearn.model_selection import StratifiedShuffleSplit
# from torch.utils.data_2025_07_22 import Sampler
from sklearn.model_selection import train_test_split
import pandas as pd
import random


# class CustomBatchSampler(Sampler):
#     def __init__(self, labels, batch_size, ratio):
#         """
#         labels: 样本的标签 (list or numpy array)
#         batch_size: 每个 batch 的总样本数
#         ratio: (num_class_1, num_class_0)，batch 内类别的数量比例
#         """
#         self.labels = np.array(labels)
#         self.batch_size = batch_size
#         self.num_class_1, self.num_class_0 = ratio
#
#         # 获取类别的索引
#         self.class_1_idx = np.where(self.labels == 1)[0]
#         self.class_0_idx = np.where(self.labels == 0)[0]
#
#     def __iter__(self):
#         # 每次迭代生成一个 batch
#         class_1_perm = np.random.permutation(self.class_1_idx)
#         class_0_perm = np.random.permutation(self.class_0_idx)
#
#         # 根据比例抽样
#         for i in range(0, len(class_1_perm), self.num_class_1):
#             batch_class_1 = class_1_perm[i:i + self.num_class_1]
#             batch_class_0 = class_0_perm[i * self.num_class_0:(i + 1) * self.num_class_0]
#
#             if len(batch_class_0) < self.num_class_0 or len(batch_class_1) < self.num_class_1:
#                 break  # 防止越界
#
#             # 合并两个类别的样本并打乱顺序
#             batch_indices = np.concatenate([batch_class_1, batch_class_0])
#             np.random.shuffle(batch_indices)
#             yield batch_indices
#
#     def __len__(self):
#         # 计算 batch 的总数（按类别 1 的样本数量限制）
#         return len(self.class_1_idx) // self.num_class_1

class CustomDataset(Dataset):
    def __init__(self, virus_data, host_data, target_data, similarity=None):
        """
        Initialize the dataset.
        :param virus_data: Tensor of virus-related features.
        :param host_data: Tensor of host-related features.
        :param target_data: Tensor of target labels.
        """
        self.virus_data = virus_data
        self.host_data = host_data
        self.target_data = target_data
        self.similarity = similarity

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        """
        Return one sample of data_2025_07_22 at the given index.
        """
        if self.similarity is not None:
            return self.virus_data[idx], self.host_data[idx], self.target_data[idx], idx, self.similarity
        else:
            return self.virus_data[idx], self.host_data[idx], self.target_data[idx], idx, 0
        # return self.virus_data[idx], self.host_data[idx], self.target_data[idx], idx


def create_dataloaders(virus_data, host_data, target_data, simi_data=None, batch_size=32, val_split=0.2,
                       test_split=0.1):
    """
    Create DataLoaders for training, validation, and testing.
    :param virus_data: Tensor of virus-related features.
    :param host_data: Tensor of host-related features.
    :param target_data: Tensor of target labels.
    :param simi_data: Tensor of target labels.
    :param batch_size: Batch size for the DataLoader.
    :param val_split: Fraction of the dataset to use for validation.
    :param test_split: Fraction of the dataset to use for testing.
    :return: DataLoaders for training, validation, and testing.
    """
    # # Convert input data_2025_07_22 to a dataset
    # # target_data = np.array(target_data).flatten()
    # # virus_data = torch.concat([virus_data,simi_data],dim=-1)
    # # dataset = CustomDataset(virus_data, host_data,target_data,simi_data)
    # dataset = CustomDataset(simi_data, host_data, target_data, None)
    #
    # labels = target_data.numpy() if isinstance(target_data, torch.Tensor) else target_data
    #
    # # # Further split train+val set into train and val sets
    # # strat_split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split / (1 - test_split), random_state=42)
    # #     # for train_idx, val_idx in strat_split_val.split(train_val_idx, labels[train_val_idx]):
    # #     train_idx = list(train_idx)  # Training set indices
    # #     val_idx = list(val_idx)  # Validation set indices
    #
    # # Stratified split using sklearn
    # strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
    # for train_idx, test_idx in strat_split.split(range(len(dataset)), labels):
    #     train_idx = list(train_idx)  # Training set indices
    #     test_idx = list(test_idx)  # Test set indices
    # # train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_split, random_state=42)
    #
    # #
    # labels = np.array(labels).flatten()
    # labels_int = np.array(labels).astype(int)
    # # print((labels[train_idx] == 1).sum())
    # # print((labels[train_idx] == 1).sum() + (labels[test_idx] == 1).sum())
    #
    # class_counts = np.bincount(labels_int)  # 统计每个类别的样本数量
    # weights = 1.0 / class_counts  # 每个类别的权重（样本少的类别权重大）
    # # weights = (len(labels_int) / class_counts) #/ np.sum(1.0 / class_counts)
    #
    # sample_weights = weights[labels_int]  # 为每个样本赋权重
    #
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights[train_idx],
    #     num_samples=len(train_idx),  # 使用训练集的样本数
    #     replacement=True
    # )
    # sampler_idx = []
    # for idx in sampler:
    #     sampler_idx.append(idx)
    # # # 创建 WeightedRandomSampler
    # # # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
    # from collections import Counter
    #
    # counted_idx = Counter(sampler_idx)
    #
    # # 查找重复的 idx（出现次数大于 1）
    # repeated_idx = {idx: count for idx, count in counted_idx.items() if count > 1}
    # # print(f"重复的 idx: {repeated_idx}")
    #
    # # # 创建 DataLoader
    # train_subset = Subset(dataset, train_idx)
    # # sampler = CustomBatchSampler(labels_int, batch_size=batch_size, ratio=((labels_int == 1).sum().item(), (labels_int == 0).sum().item()))
    # # train_loader = DataLoader(dataset, batch_sampler=sampler)
    #
    # # train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    # #
    # train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    # # train_loader = DataLoader(ConcatDataset([Subset(dataset, test_idx), Subset(dataset, train_idx)]), batch_size=batch_size,
    # #                         shuffle=True)
    # # sampler_test = WeightedRandomSampler(
    # #     weights=sample_weights[test_idx],
    # #     num_samples=len(test_idx),  # 使用训练集的样本数
    # #     replacement=True
    # # )
    # # test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, sampler=sampler_test)
    # half_idx = len(test_idx) // 2
    # # test_loader = DataLoader(Subset(dataset, test_idx[:half_idx]), batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    virus_data_valid = pd.read_csv("./data_VAE/SARS_viruses_vector.csv", index_col=0, header=None)
    coexistence_data = pd.read_csv("./data_VAE/SARS_association_vector.csv", index_col=0, header=None)
    host_data_valid = pd.read_csv("./data_VAE/SARS_hosts_vector.csv", index_col=0, header=None)

    # virus_data_vaild_hamming = pd.read_csv("./data_VAE/SARS_hamming_vector.csv", index_col=0, header=None)
    virus_data_vaild_hamming = pd.read_csv("./data_2025_03_19_new_hamming/SARS_hamming_vector.csv", index_col=0,
                                           header=None)

    virus_data_vaild_hamming = np.array(virus_data_vaild_hamming)
    # virus_data_vaild_hamming = torch.tensor(virus_data_vaild_hamming, dtype=torch.int64)
    virus_data_vaild_hamming = torch.tensor(virus_data_vaild_hamming, dtype=torch.float32)

    # virus_data_vaild_hamming = 1 - (virus_data_vaild_hamming - torch.min(virus_data_vaild_hamming)) / (torch.max(virus_data_vaild_hamming) - torch.min(virus_data_vaild_hamming))

    virus_data_valid = np.array(virus_data_valid, dtype=np.float32)
    host_data_valid = np.array(host_data_valid, dtype=np.float32)
    virus_data_valid = torch.tensor(np.array(virus_data_valid), dtype=torch.long)
    host_data_valid = torch.tensor(host_data_valid, dtype=torch.float32)

    # virus_data_valid = torch.concat([virus_data_valid,virus_data_vaild_hamming],dim=-1)
    coexistence_data = torch.tensor(np.array(coexistence_data, dtype=np.float32), dtype=torch.float32)
    # val_dataset = CustomDataset(virus_data_valid, host_data_valid, coexistence_data,virus_data_vaild_hamming)
    val_dataset = CustomDataset(virus_data_vaild_hamming, host_data_valid, coexistence_data, None)

    # val_loader = DataLoader(ConcatDataset([Subset(dataset, test_idx[half_idx:]), val_dataset]), batch_size=batch_size,
    #                         shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train_val_idx, test_idx = train_test_split(list(range(len(target_data))), test_size=test_split, random_state=42,shuffle=True)
    #
    #
    # # Create DataLoaders
    # train_loader = DataLoader(Subset(dataset, train_val_idx), batch_size=batch_size, shuffle=False)
    # # # val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    # coexistence_data = pd.read_csv("./data_VAE/association_vector.csv", index_col=0, header=None)
    # coexistence_data = pd.read_csv("./data_repeat/association_fold_vector.csv", index_col=0, header=None)

    # sampler_idx = []
    # train_indices = train_subset.indices  # 获取 Subset 的实际索引
    # sampler_indices = list(sampler)  # 获取 WeightedRandomSampler 中的所有索引
    # virus_data_look = []
    # for batch_idx, (virus_data, host_data, target, idx_train) in enumerate(train_loader):
    #     # 这里 data_2025_07_22 是输入数据，labels 是对应标签
    #     batch_indices = [train_indices[idx] for idx in
    #                      sampler_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]]
    #     sampler_idx.append(idx_train)
    #     # print(f"Batch {batch_idx}: Sample indices {batch_indices}")
    #     # for i, index in enumerate(batch_indices):
    #     #     # 对比 virus_data[i] 和原始数据中 train_subset[index] 是否一致
    #     #     original_data = train_subset[index][0]  # 假设 train_subset 是 (data_2025_07_22, label) 形式
    #     #     assert torch.equal(virus_data[i], original_data), f"Mismatch at index {index}"
    #     # print("Batch data_2025_07_22 matches sampler indices!")
    #     # if batch_idx == 0:
    #     # print(virus_data[2], train_subset[24][0], virus_data[1845])
    #     virus_data_look.append(virus_data)
    #
    # sampler_idx = np.concatenate(sampler_idx)
    # data_index = coexistence_data.index.tolist()
    # coexistence_data = np.array(coexistence_data)
    # train_labels = [coexistence_data[i] for i in sampler_idx]
    # # 创建 DataFrame 并保存为 CSV
    # test_labels_df = pd.DataFrame(
    #     np.array(train_labels),
    #     index=[data_index[i] for i in sampler_idx],  # 使用 coexistence_data.index 对应的索引
    #     columns=["Label"]
    # )
    # # 保存为 CSV 文件
    # test_labels_df.to_csv("./data_VAE/sampler_idx_labels.csv", index=True, header=True)
    # train_loader = DataLoader(ConcatDataset([Subset(dataset, test_idx), Subset(dataset, train_idx)]), batch_size=batch_size,
    #                         shuffle=True)

    #2025.03.19 new manual split
    new_train_319_virus = pd.read_csv('./data_2025_07_22/train_vv_e.csv', header=None, index_col=0)
    new_train_319_host = pd.read_csv('./data_2025_07_22/train_hh_e.csv', header=None, index_col=0)
    new_train_319_target = pd.read_csv('./data_2025_07_22/train_vh_e.csv', header=None, index_col=0)

    new_train_319_virus = np.array(new_train_319_virus, dtype=np.float32)
    new_train_319_virus = torch.tensor(new_train_319_virus, dtype=torch.float32)
    new_train_319_host = np.array(new_train_319_host)
    new_train_319_host = torch.tensor(new_train_319_host, dtype=torch.float32)
    new_train_319_target = np.array(new_train_319_target)
    new_train_319_target = torch.tensor(new_train_319_target, dtype=torch.float32)

    new_test_319_virus = pd.read_csv('./data_2025_07_22/valid_vv_e.csv', header=None, index_col=0)
    new_test_319_host = pd.read_csv('./data_2025_07_22/valid_hh_e.csv', header=None, index_col=0)
    new_test_319_target = pd.read_csv('./data_2025_07_22/valid_vh_e.csv', header=None, index_col=0)

    new_test_319_virus = np.array(new_test_319_virus)
    new_test_319_virus = torch.tensor(new_test_319_virus, dtype=torch.float32)
    new_test_319_host = np.array(new_test_319_host)
    new_test_319_host = torch.tensor(new_test_319_host, dtype=torch.float32)
    new_test_319_target = np.array(new_test_319_target)
    new_test_319_target = torch.tensor(new_test_319_target, dtype=torch.float32)

    train_dataset = CustomDataset(new_train_319_virus, new_train_319_host, new_train_319_target, None)
    test_dataset = CustomDataset(new_test_319_virus, new_test_319_host, new_test_319_target, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
    # return train_loader, val_loader, test_loader
