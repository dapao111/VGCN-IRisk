# trainer.py
import torch
import torch.optim as optim
from model.network import VirusHostCoexistenceModel
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
import os
from sklearn.metrics import roc_curve, auc, accuracy_score, r2_score, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score
import pandas as pd
from utils.config import get_config
from model.function import FocalLoss
from sklearn.metrics import confusion_matrix

import pycuda.autoprimaryctx

import torch
import random
import numpy as np

import time

seed = int(time.time() / 333)
# seed = 4412
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class Trainer(object):
    def __init__(self, save_path: str = "", train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 **kwargs):

        # Setup config
        self._config = get_config(**kwargs)
        self.train_data = train_dataloader

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        self._config.amp = torch.cuda.is_available() and self._config.with_cuda
        if self._config.with_cuda and torch.cuda.device_count() < 1:
            print("No detected GPU device. Load the model on CPU")
            self._config.with_cuda = False
        print("The model is loaded on %s" % ("GPU" if self._config.with_cuda else "CPU"))
        self.device = torch.device("cuda:0" if self._config.with_cuda else "cpu")
        # self.device = torch.device("cpu")

        self.test_data = test_dataloader

        # To save the best model
        self.min_loss = np.inf
        self.save_path = save_path
        self.save_model_path = save_path + "model"
        if save_path and not os.path.exists(save_path):
            os.mkdir(save_path)
        self.train_res = os.path.join(self.save_path, "train.csv")
        self.eval_res = os.path.join(self.save_path, "eval.csv")
        self.csv_path = os.path.join(self.save_path, 'training_loss.csv')
        self.fold_result = os.path.join(self.save_path, 'fold_result.csv')
        self.att_result = os.path.join(self.save_path, 'att_result.csv')
        self.yamt_result = os.path.join(self.save_path, 'ymat_result.csv')
        self.ylabel_result = os.path.join(self.save_path, 'ylabel_result.csv')

        self._config.lr = 9e-5

    def save(self, file_path="res/model"):

        self.model.to("cpu")
        self.model.save_pretrained(file_path)
        self.model.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)

    def _setup_model(self, pos_weight, pretrain_file=None):

        self.model = self.model.to(self.device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if self._config.with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUs for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model)

        self.optim = optim.RMSprop(self.model.parameters(),
                                   lr=self._config.lr,
                                   alpha=self._config.alpha,  # 平滑因子，通常在0.9到0.99之间
                                   eps=self._config.eps,
                                   weight_decay=self._config.weight_decay)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()

    def to_vertex_weights(self, attention_list):
        edge_index = attention_list[0]
        edge_weights = attention_list[1].mean(-1)
        num_nodes = edge_index.max() + 1  # 节点数量
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        # 填充邻接矩阵
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            adjacency_matrix[src, dst] = edge_weights[i]
        return adjacency_matrix

    def train_model(self, virus_data, host_data, coexistence_data, virus_edge_index, host_edge_index,
                    coexistence_edge_indexs, target, virus_edge_weight, host_edge_weight, num_epochs=1, fold_=1,
                    idx=None):
        attention_weights = []
        self.model.train()
        global_step_loss = 0
        self.step = 0
        self.model = self.model.to(self.device)
        for epoch in range(num_epochs):
            self.optim.zero_grad()
            (virus_data, host_data, coexistence_data, virus_edge_index, host_edge_index,
             coexistence_edge_index, coexistence_edge_index_t, target, virus_edge_weight, host_edge_weight) = (
                virus_data.to(self.device), host_data.to(self.device), coexistence_data.to(self.device),
                virus_edge_index.to(self.device), host_edge_index.to(self.device),
                coexistence_edge_indexs[0].to(self.device), coexistence_edge_indexs[1].to(self.device),
                target.to(self.device),
                virus_edge_weight.to(self.device), host_edge_weight.to(self.device))

            # Forward pass
            output = self.model(virus_data, host_data, coexistence_data, virus_edge_index, host_edge_index,
                                coexistence_edge_index, coexistence_edge_index_t, virus_edge_weight, host_edge_weight)

            # Compute loss
            N = target.shape[0]
            if idx is not None:
                all_indices = torch.tensor(idx)
            else:
                all_indices = torch.arange(N)
            start_idx = fold_ * N // 20
            end_idx = (fold_ + 1) * N // 20
            deleted_indices = all_indices[start_idx:end_idx]
            # print(target[deleted_indices].sum())
            remaining_indices = torch.cat((all_indices[:start_idx], all_indices[end_idx:]), dim=0)
            output_filtered1 = output[0][:-6]
            output_filtered2 = output[1][:-6]
            output_filtered3 = output[2][:-6]
            # output_filtered4 = output[3][:20]
            # target[target == 2] = 1
            target_filterd = target[:-6]
            loss = (self.loss_fn(output[0], target) + self.loss_fn(output[1], target) + self.loss_fn(output[2],
                                                                                                     target)) / 3
            # loss = self.loss_fn(-output_filtered1, target_filterd)  #+ self.loss_fn(output_filtered2, target_filterd)
            # loss = (self.loss_fn(output_filtered1, target_filterd) + self.loss_fn(output_filtered2,
            #                                                                       target_filterd) + self.loss_fn(
            #     output_filtered3, target_filterd))
            # loss = loss / 3
            if epoch % self._config.eval_freq == 0:
                if self.min_loss > loss:
                    print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s" % (
                        epoch, global_step_loss, self.min_loss, self.save_model_path))
                    self.save(self.save_model_path)
                    self.min_loss = global_step_loss
                with open(self.csv_path, 'a') as f:
                    f.write(f'{epoch + 1},{loss.item():.6f}\n')
            loss.backward()
            self.step = epoch
            self.optim.step()
            attention_weights = [output[3], output[4]]
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        return (output[0] + output[1] + output[2]) / 3, attention_weights
        return output[0]

    def create_model(self, input_dim_virus, input_dim_host, hidden_dim=128, transformer_dim=256, output_dim=256,
                     pool_size=128):
        self.model = VirusHostCoexistenceModel(input_dim_virus, input_dim_host,
                                               hidden_dim, transformer_dim, output_dim, pool_size)

        pass

    def _acc(self, pred, label):

        if type(pred).__module__ != np.__name__:
            pred = pred.numpy()
        if type(label).__module__ != np.__name__:
            label = label.numpy()

        if len(pred.shape) > 1:
            pred = pred.flatten()
        if len(label.shape) > 1:
            label = label.flatten()

        return accuracy_score(y_true=label, y_pred=pred)

    def find_threshold_for_tp_greater_than_09(ymat, data):
        thresholds = np.arange(0, 1.01, 0.01)  # 从 0 到 1 的阈值
        best_threshold = None

        for threshold in thresholds:
            y_pred = np.where(ymat >= threshold, 1, 0)

            tn, fp, fn, tp = confusion_matrix(data, y_pred).ravel()

            if tp > 0.9:
                best_threshold = threshold
                break  # 找到第一个使 TP > 0.9 的阈值后退出循环

        return best_threshold

    def cal_auc(self, ymat, data, fold_number):
        y_mat_df = pd.DataFrame(ymat)
        y_mat_df.to_csv(self.yamt_result, mode='a', index=False, header=False)
        y_label_df = pd.DataFrame(data)
        y_label_df.to_csv(self.ylabel_result, mode='a', index=False, header=False)

        ymat = ymat.flatten()
        data = data.flatten()

        # 计算 AUROC 和 AUPR
        fpr, tpr, rocth = roc_curve(data, ymat)
        auroc = auc(fpr, tpr)

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
        # y_pred_07 = np.where(np.array(ymat) >= 0.75, 1, 0)
        # y_pred_098 = np.where(np.array(ymat) >= 0.98, 1, 0)
        # 将结果记录到 CSV 文件
        results = pd.DataFrame({
            'Fold': [fold_number],
            'AUROC': [auroc],
            'AUPR': [aupr],
            'TSS': [tss],  # 添加 TSS
            'precision_mean': [precisions],
            'recall_mean': [recall_1],
            'f1_score_mean': [f1_score_],
            'precision_0.5': [precision],
            'recall_0.5': [recall],
            'f1_score_0.5': [f1_s],
            'seed': seed,
        })

        # 将结果保存到 CSV 文件
        results.to_csv(self.fold_result, mode='a', header=not os.path.exists(self.fold_result), index=False)

        print('AUROC= %.4f | AUPR= %.4f | TSS= %.4f | precision= %.4f | recall= %.4f | f1_score= %.4f' % (
            auroc, aupr, tss, precision, recall, f1_s))

        return auroc, aupr

    def model_runner(self, virus_data, host_data, coexistence_data, virus_edge_index, host_edge_index,
                     coexistence_edge_index, target, virus_edge_weight, host_edge_weight, to_valid_data_num=0,
                     num_epochs=2000, fold_cv=5):
        N = target.shape[0] - to_valid_data_num  #464
        random_N = N // fold_cv
        # idx = np.arange(N - random_N)
        # idx = np.random.shuffle(idx)
        # # 获取最后一个 fold 的索引范围并保持不变
        # last_fold_idx = np.arange(N - random_N, N)
        # # 将打乱的索引与最后一个 fold 的索引合并
        # idx = np.concatenate([idx, last_fold_idx])
        res = torch.zeros(fold_cv, target.shape[0] - to_valid_data_num, target.shape[1])
        aurocl = np.zeros(fold_cv)
        auprl = np.zeros(fold_cv)
        attention_weights_all = []
        # print(target[1:50].sum())
        # random_idx = np.random.permutation(target.shape[0] - random_N)
        random_idx = np.random.permutation(target.shape[0])
        # last_fold_idx = np.arange(N - random_N, N)
        # # 将打乱的索引与最后一个 fold 的索引合并
        # random_idx = np.concatenate([random_idx, last_fold_idx])
        row_indices = []
        col_indices = []
        # for i in range(len(random_idx)):
        #     for j in range(i + 1, len(random_idx)):
        #         row_indices.append(random_idx[i])
        #         col_indices.append(random_idx[j])
        # coexistence_edge_index00 = coexistence_edge_index.numpy()
        # coexistence_edge_index00[0] = random_idx[coexistence_edge_index00[0]]
        # coexistence_edge_index00[1] = random_idx[coexistence_edge_index00[1]]
        # coexistence_edge_index00 = torch.tensor(coexistence_edge_index00)
        # virus_edge_index = coexistence_edge_index00
        # coexistence_edge_index = virus_edge_index
        # target = target[random_idx]
        idx = np.arange(N)
        # virus_data = virus_data[random_idx]
        # virus_data = virus_data[:, random_idx]

        # virus_data = virus_data[idx]
        np.random.shuffle(idx)
        # virus_data = virus_data[idx]
        # target = target[idx]
        to_valid_virus_data = virus_data
        if to_valid_data_num > 0:
            to_valid_virus_data = virus_data[-to_valid_data_num:, :-to_valid_data_num]
            virus_data = virus_data[:-to_valid_data_num, :-to_valid_data_num]
        final_target = target
        if to_valid_data_num > 0:
            target = target[:-to_valid_data_num, :]
            coexistence_data = coexistence_data[:-to_valid_data_num, :]
        for i in range(fold_cv):
            self.create_model(target.shape[0], target.shape[1])
            self._setup_model((target == 0).sum().item() / (target == 1).sum().item())
            print("Fold {}".format(i + 1))
            A0 = target.clone()
            # for j in range(i * N // fold_cv, (i + 1) * N // fold_cv):
            #     A0[j, :] = torch.full((target.shape[1], 1), 0.5).view(-1)
            A0[(i * N // fold_cv):(i + 1) * (N // fold_cv)] = 0.5
            # A0[-6:] = 0.5
            # target_1 = A0
            # for j in range(i * N // fold_cv, (i + 1) * N // fold_cv):
            #     A0[idx[j], :] = torch.full((target.shape[1],1),0).view(-1)
            # target_2 = A0
            target_ = A0

            resi, attention_weights = self.train_model(virus_data, host_data, target_, virus_edge_index,
                                                       host_edge_index, coexistence_edge_index, coexistence_data,
                                                       virus_edge_weight, host_edge_weight, num_epochs, i)
            # resi = scaley(resi)
            # resi = (resi - torch.min(resi)) / (torch.max(resi) - torch.min(resi))
            resi = torch.sigmoid(resi)
            count_greater_than_09 = (resi > 0.9821).sum().item()
            print(count_greater_than_09)
            res[i] = resi

            attention_weights_all.append(
                [self.to_vertex_weights(attention_weights[0]), self.to_vertex_weights(attention_weights[1])])

            if self._config.with_cuda:
                resi = resi.cpu().detach().numpy()
            else:
                resi = resi.detach().numpy()

            # auroc, aupr = self.cal_auc(resi, target, i)
            # auroc_val, aupr_val = self.cal_auc(resi[-6:], target[-6:], i)
            auroc_val, aupr_val = self.cal_auc(resi[(i * N // fold_cv):(i + 1) * (N // fold_cv)], target[(i * N // fold_cv):(i + 1) * (N // fold_cv)], i)

            aurocl[i] = auroc_val
            auprl[i] = aupr_val
            # with open(self.csv_path, 'a') as f:
            #     f.write(f'{epoch + 1},{loss.item():.6f}\n')
        ymat = res[aurocl.argmax()].detach().numpy()
        print("===Final result===")
        print('AUROC= %.4f +- %.4f | AUPR= %.4f +- %.4f' % (aurocl.mean(), aurocl.std(), auprl.mean(), auprl.std()))
        # auroc, aupr = self.cal_auc(ymat, target, fold_cv )
        # print("===predict result===")
        # print('AUROC= %.4f | AUPR= %.4f' % (auroc, aupr))
        for weights in attention_weights_all:
            print(weights[0], type(weights[0]))
        # if self._config.with_cuda:
        #     virus_weights_all = [weights[0].cpu().detach().numpy() for weights in attention_weights_all]
        #     host_weights_all = [weights[1].cpu().detach().numpy() for weights in attention_weights_all]
        # else:
        #     virus_weights_all = [weights[0].detach().numpy() for weights in attention_weights_all]
        #     host_weights_all = [weights[1].detach().numpy() for weights in attention_weights_all]
        virus_weights_all = [weights[0] for weights in attention_weights_all]
        host_weights_all = [weights[1] for weights in attention_weights_all]
        host_weights_final = np.mean(host_weights_all, axis=0)
        virus_weights_final = np.mean(virus_weights_all, axis=0)

        pd.DataFrame(host_weights_final).to_csv(self.save_path + 'host_weights_final.csv', index=False)
        pd.DataFrame(virus_weights_final).to_csv(self.save_path + 'virus_weights_final.csv', index=False)
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 绘制热力图并保存
        # plt.figure(figsize=(10, 6))
        sns.heatmap(host_weights_final, cmap='viridis')
        plt.title('Host Weights Heatmap')
        plt.savefig(self.save_path + 'host_weights_heatmap.png')
        plt.close()

        # plt.figure(figsize=(10, 6))
        sns.heatmap(virus_weights_final, cmap='viridis')
        plt.title('Virus Weights Heatmap')
        plt.savefig(self.save_path + 'virus_weights_heatmap.png')
        plt.close()
        # if self._config.with_cuda:
        #     return virus_weights_final.cpu().detach().numpy(),
        # else:
        #     return attention_weights_all.detach().numpy()
        # to_valid_virus_data = torch.vstack((virus_data, to_valid_virus_data))
        # to_valid_virus_data = to_valid_virus_data.to(self.device)
        # host_data, coexistence_data, virus_edge_index, host_edge_index, coexistence_edge_index = (
        #     host_data.to(self.device), coexistence_data.to(self.device),
        #     virus_edge_index.to(self.device), host_edge_index.to(self.device), coexistence_edge_index.to(self.device))
        # valid_output1, valid_output2 = self.model(to_valid_virus_data, host_data, coexistence_data, virus_edge_index,
        #                                           host_edge_index,
        #                                           coexistence_edge_index)
        # if self._config.with_cuda:
        #     valid_output1 = valid_output1.cpu().detach().numpy()
        # else:
        #     valid_output1 = valid_output1.detach().numpy()
        # auroc, aupr = self.cal_auc(valid_output1[-to_valid_data_num:], final_target[-to_valid_data_num:], fold_cv)
        # print("===predict result===")
        # print('AUROC= %.4f | AUPR= %.4f' % (auroc, aupr))
