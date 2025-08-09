# trainer.py
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
# from model.network_VAE import VAE_transmission_model
# from model.network_VAE_decoderconc import VAE_transmission_model
from model.network_class_ori import VAE_transmission_model, EnhancedVAE
from torch.optim.swa_utils import AveragedModel, SWALR

from torch.autograd import Variable, grad

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

# import pycuda.autoprimaryctx

import torch
import random
import numpy as np

import time

# seed = 5214394
# seed = 7505184
seed = int(time.time() / 333) + int(time.time() / 777)
# seed = 7467633
# seed =7467633
# seed = 5210272
# seed = 7451338
# seed = 5214394
# seed = 7466240
# seed = 7515451
# seed = 7521517
# seed = 442
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


class localLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2, alpha=0.25):
        super(localLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)  # We only need the term for the positive class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss


class Trainer(object):
    def __init__(self, save_path: str = "", train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 **kwargs):

        # Setup config
        self._config = get_config(**kwargs)
        self.train_data = train_dataloader
        self.val_data = val_dataloader

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
        self.max_tss = -1

        self.save_path = save_path
        self.save_model_path = os.path.join(self.save_path, 'model')
        self.save_tss_model_path = os.path.join(self.save_path, 'tss_model')
        self.save_data_label_path = os.path.join(self.save_path, 'idx_label')

        if save_path and not os.path.exists(save_path):
            os.mkdir(save_path)
        if self.save_data_label_path and not os.path.exists(self.save_data_label_path):
            os.mkdir(self.save_data_label_path)
        self.train_res = os.path.join(self.save_path, "train.csv")
        self.eval_res = os.path.join(self.save_path, "eval.csv")
        self.csv_path = os.path.join(self.save_path, 'training_loss.csv')
        self.fold_result = os.path.join(self.save_path, 'fold_result.csv')
        self.att_result = os.path.join(self.save_path, 'att_result.csv')
        self.yamt_result = os.path.join(self.save_path, 'ymat_result.csv')
        self.yamt_eval_result = os.path.join(self.save_path, 'ymat_eval_result.csv')
        self.yamt_sas_eval_result = os.path.join(self.save_path, 'ymat_sas_eval_result.csv')

        self.ylabel_result = os.path.join(self.save_path, 'ylabel_result.csv')
        self.train_ymat_result = os.path.join(self.save_path, 'train_ymat_result.csv')

        self._config.lr = 3e-5

        set_seed(seed)

    def save(self, file_path="res/model"):

        # self.model.to("cpu")
        # self.model.save_pretrained(file_path, self.step)
        os.makedirs(file_path, exist_ok=True)
        model_weights_path = os.path.join(file_path, f'pytorch_model{self.step}.pth')

        # # 获取要保存的模型状态
        # if hasattr(self.model, 'module'):
        #     # DataParallel包装的模型
        #     state_dict = self.model.module.state_dict()
        # else:
        #     # 普通模型
        state_dict = self.model.state_dict()

        # 保存模型权重
        torch.save(state_dict, model_weights_path)
        self.last_step = self.step
        self.model.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)

    def custom_update_bn(self, loader, swa_model, device):
        """
        自定义的BatchNorm更新函数，适配多输入模型
        """
        swa_model.train()
        with torch.no_grad():
            for batch in loader:
                # 假设loader返回的是 (virus_data, host_data)
                virus_data = batch[0].to(device)
                host_data = batch[1].to(device)
                noise = self.make_noise(virus_data.shape[0], 10)
                swa_model(virus_data, host_data, noise)  # 显式传递两个参数

    def kl_loss(self, mu, log_var):
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def _setup_model(self, pos_weight, pretrain_file=None):

        self.model = self.model.to(self.device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if self._config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.optim = optim.RMSprop(self.model.parameters(),
                                   lr=self._config.lr,
                                   alpha=self._config.alpha,  # 平滑因子，通常在0.9到0.99之间
                                   eps=self._config.eps,
                                   weight_decay=self._config.weight_decay)

        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optim, swa_lr=1e-4)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        # self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.NLLLoss()
        pos_weight = torch.tensor([1.0, pos_weight], device=self.device)
        # self.loss_fn = nn.CrossEntropyLoss(weight=pos_weight)
        # self.loss_fn = nn.CrossEntropyLoss(weight=pos_weight)

        self.recons_loss_fn = nn.MSELoss(reduction="mean")
        # self.loss_fn = localLoss(pos_weight)
        # self.VAE_KL_loss_fn = self.kl_loss
        self.GB_loss_fn = nn.GaussianNLLLoss(eps=1e-4)
        self.VAE_KL_loss_fn = self.kl_loss

    def train_model(self, virus_data, host_data, target, num_epochs=1, fold_=1, idx=None):
        attention_weights = []
        self.model.train()
        global_step_loss = 0
        self.step = 0
        self.last_step = -1
        self.model = self.model.to(self.device)
        loss_total = []
        for epoch in range(num_epochs):
            self.optim.zero_grad()
            virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                self.device),

            # Forward pass
            output = self.model(virus_data, host_data)

            loss = self.loss_fn(output, target)
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            loss_total.append(loss.item())
        plt.plot(range(1, num_epochs + 1), loss_total, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(self.save_path + 'loss.png')
        plt.close()
        return output, attention_weights

    def create_model(self, input_dim_virus, input_dim_host, max_frequence, hidden_dim=128):
        # self.model = VAE_transmission_model(input_dim_virus, input_dim_host, hidden_dim, max_frequence, embed_dim=1024)
        self.model = EnhancedVAE(input_dim_virus, input_dim_host, max_frequence)
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

    def cal_auc(self, ymat, data, fold_number):
        y_mat_df = pd.DataFrame(ymat)
        if fold_number == 'sas':
            y_mat_df.to_csv(self.yamt_sas_eval_result, mode='a', index=False, header=False)
        else:
            y_mat_df.to_csv(self.yamt_eval_result, mode='a', index=False, header=False)
        y_label_df = pd.DataFrame(data)
        y_label_df.to_csv(self.ylabel_result, mode='a', index=False, header=False)
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
        if fold_number == 'sas':
            plt.savefig(self.save_path + 'sas_roc.pdf', format='pdf')
        else:
            plt.savefig(self.save_path + 'roc.pdf', format='pdf')

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
        if fold_number == 'sas':
            plt.savefig(self.save_path + 'sas_fpr-precision_xticks_eval.pdf', format='pdf')
        else:
            plt.savefig(self.save_path + 'fpr-precision_xticks_eval.pdf', format='pdf')

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
        print('eval res:\n')

        print('AUROC= %.4f | AUPR= %.4f | TSS= %.4f | precision= %.4f | recall= %.4f | f1_score= %.4f' % (
            auroc, aupr, tss, precision, recall, f1_s))

        return auroc, aupr

    def cal_auc_train_last(self, ymat, data, fold_number):
        y_mat_df = pd.DataFrame(ymat)
        y_mat_df.to_csv(self.train_ymat_result, mode='a', index=False, header=False)
        # y_label_df = pd.DataFrame(data_2025_07_22)
        # y_label_df.to_csv(self.ylabel_result, mode='a', index=False, header=False)
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
        plt.savefig(self.save_path + 'roc_training_set.pdf', format='pdf')
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
        plt.savefig(self.save_path + 'fpr-precision_training_set.pdf', format='pdf')
        plt.close()
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
        # results.to_csv(self.fold_result, mode='a', header=not os.path.exists(self.fold_result), index=False)

        print('AUROC= %.4f | AUPR= %.4f | TSS= %.4f | precision= %.4f | recall= %.4f | f1_score= %.4f' % (
            auroc, aupr, tss, precision, recall, f1_s))

        return auroc, aupr

    def cal_auc_epoch(self, ymat, data, epoch, test_data=None):

        y_mat_df = pd.DataFrame(ymat)
        # y_mat_df.to_csv(self.yamt_result, mode='a', index=False, header=False)
        y_label_df = pd.DataFrame(data)
        # y_label_df.to_csv(self.ylabel_result, mode='a', index=False, header=False)
        ymat = ymat.flatten()
        data = data.flatten()

        # 计算 AUROC 和 AUPR
        fpr, tpr, rocth = roc_curve(data, ymat)
        auroc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.2f})')
        # plt.plot([0, 1], [0, 1], 'k--')  # 画对角线，表示随机猜测
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc='lower right')
        # plt.savefig(self.save_path + 'roc.pdf', format='pdf')
        # plt.close()
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
        results = pd.DataFrame({
            'Fold': [epoch],
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
        if test_data == 'test':
            results.to_csv(self.fold_result, mode='a', header=not os.path.exists(self.fold_result), index=False)

        print('AUROC= %.4f | AUPR= %.4f | TSS= %.4f | precision= %.4f | recall= %.4f | f1_score= %.4f' % (
            auroc, aupr, tss, precision, recall, f1_s))

        return auroc, aupr, tss

    def make_noise(self, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(self.device, dtype=torch.float32)
        return noise

    def model_runner(self, virus_data, host_data, coexistence_data, target, to_valid_data_num=0, num_epochs=1000):

        self.model.train()
        self.step = 0
        self.model = self.model.to(self.device)
        loss_total = []

        for epoch_ in range(num_epochs):
            self.model.train()
            epoch_res = []
            loss_epcoh = 0
            target_epoch = []

            for batchidx, (virus_data, host_data, target, idx_train, simi_data) in enumerate(self.train_data):
                self.optim.zero_grad()
                if torch.all(simi_data != 0):
                    virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                        self.device), target.to(
                        self.device), simi_data.to(self.device)
                else:
                    virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                        self.device)
                # print(virus_data[2],virus_data[1845])
                # print(len(target), (target == 1).sum().item())
                noise = self.make_noise(virus_data.shape[0], 10)

                # Forward pass
                output, decoded_outputs = self.model(virus_data, host_data, noise)
                gene_vars = torch.nn.functional.softplus(decoded_outputs[2][1])
                gene_vars = torch.exp(gene_vars)
                rl_loss = 1 * self.recons_loss_fn(decoded_outputs[0], decoded_outputs[1])
                VAE_loss = self.VAE_KL_loss_fn(decoded_outputs[2][0], decoded_outputs[2][1])
                GB_loss = self.GB_loss_fn(decoded_outputs[2][0],decoded_outputs[1], gene_vars)
                # loss = 1 * self.loss_fn(output, target.squeeze().long())+ GB_loss+VAE_loss #+ rl_loss
                # loss = 0.5*self.loss_fn(output, target.squeeze().long()) + 0.3*rl_loss + 0.2*VAE_loss
                loss = 0.5*self.loss_fn(output, target) + 0.3*rl_loss + 0.2*VAE_loss

                # print(VAE_loss, rl_loss)
                # print((torch.sigmoid(output) > 0.9821).sum().item())
                # predictions = torch.argmax(output, dim=1)  # 预测的类别索引，形状为 (batch_size,)
                epoch_res.append(torch.sigmoid(output))
                # epoch_res.append(torch.sigmoid(output[:, 1]))
                # epoch_res.append(output[:, 1])
                # epoch_res.append(output)

                target_epoch.append(target)
                loss.backward()
                self.step = epoch_
                self.optim.step()
                loss_epcoh = loss_epcoh + loss.item()
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
            print(VAE_loss, rl_loss)
            loss_epcoh = loss_epcoh / len(self.train_data)
            print(f"Epoch [{epoch_ + 1}/{num_epochs}], Loss: {loss_epcoh:.4f}")
            loss_total.append(loss_epcoh / len(self.train_data))
            if self._config.with_cuda:
                outputs = torch.cat(epoch_res, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
                targets = torch.cat(target_epoch, dim=0).detach().cpu().numpy()
            else:
                outputs = torch.cat(epoch_res, dim=0).detach().numpy()  # 拼接并转为 numpy
                targets = torch.cat(target_epoch, dim=0).detach().numpy()

            self.cal_auc_epoch(outputs, targets, epoch_)
            # if (epoch_ + 1) % self._config.eval_freq == 0:
            if (epoch_ + 1) % 50 == 0:

                # if self.min_loss > loss_epcoh:
                # print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s" % (
                #     epoch_, loss_epcoh, self.min_loss, self.save_model_path))
                self.model.eval()
                self.save(self.save_model_path)
                self.min_loss = loss_epcoh

                with open(self.csv_path, 'a') as f:
                    f.write(f'{epoch_ + 1},{loss_epcoh:.6f}\n')

                eval_outputs = []
                eval_targets = []
                self.model.eval()
                with torch.no_grad():
                    # for virus_data, host_data, target, _ in self.test_data:
                    #     virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(self.device)
                    for virus_data, host_data, target, _, simi_data in self.test_data:
                        if torch.all(simi_data != 0):
                            virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                                self.device), target.to(self.device), simi_data.to(self.device)
                        else:
                            virus_data, host_data, target = virus_data.to(self.device), host_data.to(
                                self.device), target.to(self.device)
                        noise = self.make_noise(virus_data.shape[0], 10)
                        # Forward pass
                        output, decoded_output = self.model(virus_data, host_data, noise)
                        # Collect outputs for AUC calculation
                        eval_outputs.append(torch.sigmoid(output))
                        # eval_outputs.append(torch.sigmoid(output[:, 1]))
                        # eval_outputs.append(output[:, 1])
                        # eval_outputs.append(output)
                        eval_targets.append(target)

                if self._config.with_cuda:
                    eval_outputs = torch.cat(eval_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
                    eval_targets = torch.cat(eval_targets, dim=0).detach().cpu().numpy()
                else:
                    eval_outputs = torch.cat(eval_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
                    eval_targets = torch.cat(eval_targets, dim=0).detach().numpy()
                auroc, aupr, tss = self.cal_auc_epoch(eval_outputs, eval_targets, epoch_, 'test')
                # if self.max_tss < tss:
                #     print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s" % (
                #         epoch_, loss_epcoh, self.max_tss, self.save_tss_model_path))
                #     self.save(self.save_tss_model_path)
                #     self.max_tss = tss
                # count_greater_than_09 = (train_bst_result > 0.9821).sum().item()
                # print(count_greater_than_09)
                eval_sas_outputs = []
                eval_sas_targets = []
                self.model.eval()
                with torch.no_grad():
                    for virus_data, host_data, target, eval_index, simi_data in self.val_data:
                        if torch.all(simi_data != 0):
                            virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                                self.device), target.to(
                                self.device), simi_data.to(self.device)
                        else:
                            virus_data, host_data, target = virus_data.to(self.device), host_data.to(
                                self.device), target.to(
                                self.device)
                        noise = self.make_noise(virus_data.shape[0], 10)
                        # Forward pass
                        output, decoded_output = self.model(virus_data, host_data, noise)
                        # Collect outputs for AUC calculation
                        eval_sas_outputs.append(torch.sigmoid(output))
                        # eval_sas_outputs.append(torch.sigmoid(output[:, 1]))
                        # eval_sas_outputs.append(output[:, 1])
                        # eval_sas_outputs.append(output)

                        eval_sas_targets.append(target)
                if self._config.with_cuda:
                    eval_sas_outputs = torch.cat(eval_sas_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
                    eval_sas_targets = torch.cat(eval_sas_targets, dim=0).detach().cpu().numpy()
                else:
                    eval_sas_outputs = torch.cat(eval_sas_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
                    eval_sas_targets = torch.cat(eval_sas_targets, dim=0).detach().numpy()
                self.cal_auc_epoch(eval_sas_outputs, eval_sas_targets, epoch_, 'sas_test')
                eval_sas_outputs = []
                eval_sas_targets = []
                self.model.eval()
                with torch.no_grad():
                    for virus_data, host_data, target, eval_index, simi_data in self.val_data:
                        if torch.all(simi_data != 0):
                            virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                                self.device), target.to(
                                self.device), simi_data.to(self.device)
                        else:
                            virus_data, host_data, target = virus_data.to(self.device), host_data.to(
                                self.device), target.to(
                                self.device)
                        noise = self.make_noise(virus_data.shape[0], 10)
                        # Forward pass
                        output, decoded_output = self.model(virus_data, host_data, noise)
                        # Collect outputs for AUC calculation
                        eval_sas_outputs.append(torch.sigmoid(output))
                        # eval_sas_outputs.append(torch.sigmoid(output[:, 1]))
                        # eval_sas_outputs.append(output[:, 1])
                        # eval_sas_outputs.append(output)

                        eval_sas_targets.append(target)
                if self._config.with_cuda:
                    eval_sas_outputs = torch.cat(eval_sas_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
                    eval_sas_targets = torch.cat(eval_sas_targets, dim=0).detach().cpu().numpy()
                else:
                    eval_sas_outputs = torch.cat(eval_sas_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
                    eval_sas_targets = torch.cat(eval_sas_targets, dim=0).detach().numpy()
                print("valid_sas_bodong:\n")
                self.cal_auc_epoch(eval_sas_outputs, eval_sas_targets, epoch_, 'sas_test')
        plt.plot(range(1, num_epochs + 1), loss_total, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(self.save_path + 'loss.png')
        plt.close()

        # torch.optim.swa_utils.update_bn(self.train_data, self.swa_model, device=self.device)
        self.custom_update_bn(self.train_data, self.swa_model, device=self.device)

        # 保存结果
        torch.save(self.swa_model.state_dict(), self.save_model_path + "/last_swa_model.bin")

        # self.model = self.swa_model
        self.model.eval()
        # if hasattr(self.model, 'module'):
        #     self.model = self.model.module
        # self.model.load_state_dict(torch.load(self.save_model_path + '/pytorch_model' + str(self.last_step) + '.pth'),
        #                            strict=True)
        # self.model.load_state_dict(torch.load(self.save_model_path + '/last_swa_model.bin'))

        train_val_res = []
        train_val_targets = []
        sampler_idx = []
        self.model.eval()
        with torch.no_grad():
            # for virus_data, host_data, target, idx_train in self.train_data:
            #     sampler_idx.append(idx_train)
            #     virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
            #         self.device)
            for virus_data, host_data, target, idx_train, simi_data in self.train_data:
                sampler_idx.append(idx_train)
                if torch.all(simi_data != 0):
                    virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                        self.device), target.to(
                        self.device), simi_data.to(self.device)
                else:
                    virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                        self.device)
                noise = self.make_noise(virus_data.shape[0], 10)
                # Forward pass
                output, decoded_output = self.model(virus_data, host_data, noise)
                # Collect outputs for AUC calculation
                train_val_res.append(torch.sigmoid(output))
                # train_val_res.append(output)

                # train_val_res.append(torch.sigmoid(output[:, 1]))
                # train_val_res.append(output[:, 1])
                train_val_targets.append(target)

        # coexistence_data_index = pd.read_csv("./data_VAE/association_vector.csv", index_col=0, header=None)
        # coexistence_data_index = pd.read_csv("./data_repeat/association_fold_vector.csv", index_col=0, header=None)
        # coexistence_data_index = pd.read_csv('./data_2025_03_19_new_hamming/train_0_vh.csv',header=None, index_col=0)
        coexistence_data_index = pd.read_csv('./data_2025_07_22/train_vh_e.csv', header=None, index_col=0)

        sampler_idx = np.concatenate(sampler_idx)
        data_index = coexistence_data_index.index.tolist()
        coexistence_data_index = np.array(coexistence_data_index)
        train_labels = [coexistence_data_index[i] for i in sampler_idx]
        # 创建 DataFrame 并保存为 CSV
        last_train_labels_df = pd.DataFrame(
            np.array(train_labels),
            index=[data_index[i] for i in sampler_idx],  # 使用 coexistence_data.index 对应的索引
            columns=["Label"]
        )
        # 保存为 CSV 文件
        # last_train_labels_df.to_csv("./data_repeat/2025_06_24_sampler_train_last_idx_labels.csv", index=True, header=True)
        # last_train_labels_df.to_csv(self.save_data_label_path + "/2025_06_24_sampler_train_last_idx_labels.csv", index=True,
        #                             header=True)
        last_train_labels_df.to_csv(self.save_data_label_path + "/2025_07_22_sampler_train_last_idx_labels.csv", index=True,
                                    header=True)
        if self._config.with_cuda:
            train_val_res = torch.cat(train_val_res, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            train_val_targets = torch.cat(train_val_targets, dim=0).detach().cpu().numpy()
        else:
            train_val_res = torch.cat(train_val_res, dim=0).detach().numpy()  # 拼接并转为 numpy
            train_val_targets = torch.cat(train_val_targets, dim=0).detach().numpy()

        self.cal_auc_train_last(train_val_res, train_val_targets, num_epochs)

        test_outputs = []
        test_targets = []
        idx_tests = []
        self.model.eval()
        with torch.no_grad():
            # for virus_data, host_data, target, idx_test in self.test_data:
            #     virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
            #         self.device)
            for virus_data, host_data, target, idx_test, simi_data in self.test_data:
                if torch.all(simi_data != 0):
                    virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                        self.device), target.to(
                        self.device), simi_data.to(self.device)
                else:
                    virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                        self.device)
                idx_tests.append(idx_test)
                noise = self.make_noise(virus_data.shape[0], 10)
                # Forward pass
                output, decoded_output = self.model(virus_data, host_data, noise)
                # Collect outputs for AUC calculation
                # test_outputs.append(output)
                test_outputs.append(torch.sigmoid(output))

                # test_outputs.append(torch.sigmoid(output[:, 1]))
                # test_outputs.append(output[:, 1])
                test_targets.append(target)

        self.model.eval()
        if self._config.with_cuda:
            test_outputs = torch.cat(test_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            test_targets = torch.cat(test_targets, dim=0).detach().cpu().numpy()
        else:
            test_outputs = torch.cat(test_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
            test_targets = torch.cat(test_targets, dim=0).detach().numpy()
        self.cal_auc(test_outputs, test_targets, num_epochs)

        # coexistence_data_index = pd.read_csv("./data_repeat/association_fold_vector.csv", index_col=0, header=None)
        # coexistence_data_index = pd.read_csv("./data_VAE/association_vector.csv", index_col=0, header=None)
        # coexistence_data_index = pd.read_csv('./data_2025_03_19_new_hamming/valid_0_vh.csv',header=None, index_col=0)
        coexistence_data_index = pd.read_csv('./data_2025_07_22/valid_vh_e.csv', header=None, index_col=0)

        last_test_idx = np.concatenate(idx_tests)
        data_index = coexistence_data_index.index.tolist()
        coexistence_data_index = np.array(coexistence_data_index)
        test_labels = [coexistence_data_index[i] for i in last_test_idx]
        # 创建 DataFrame 并保存为 CSV
        test_labels_df = pd.DataFrame(
            np.array(test_labels),
            index=[data_index[i] for i in last_test_idx],  # 使用 coexistence_data.index 对应的索引
            columns=["Label"]
        )
        # 保存为 CSV 文件
        test_labels_df.to_csv(self.save_data_label_path + "/test_last_idx_labels.csv", index=True, header=True)
        # test_labels_df.to_csv("./data_VAE/test_last_idx_labels.csv", index=True, header=True)

        #val loader
        eval_outputs = []
        eval_targets = []
        eval_indices = []
        self.model.eval()
        # with (torch.no_grad()):
        # for virus_data, host_data, target, eval_index in self.val_data:
        #     virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
        #         self.device)
        for virus_data, host_data, target, eval_index, simi_data in self.val_data:
            if torch.all(simi_data != 0):
                virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                    self.device), target.to(
                    self.device), simi_data.to(self.device)
            else:
                virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                    self.device)
            noise = self.make_noise(virus_data.shape[0], 10)
            # Forward pass
            output, decoded_output = self.model(virus_data, host_data, noise)
            # Collect outputs for AUC calculation
            eval_outputs.append(torch.sigmoid(output))
            # eval_outputs.append(output)
            # eval_outputs.append(torch.sigmoid(output[:, 1]))
            # eval_outputs.append((output[:, 1]))

            eval_targets.append(target)
            eval_indices.append(eval_index)

        coexistence_data_sas_index = pd.read_csv("./data_VAE/SARS_association_vector.csv", index_col=0, header=None)
        eval_indices_last = np.concatenate(eval_indices)
        self.model.eval()
        # eval_indices = np.array(eval_index)
        data_index = coexistence_data_sas_index.index.tolist()
        coexistence_data_sas_index = np.array(coexistence_data_sas_index)
        sas_labels = [coexistence_data_sas_index[i] for i in eval_indices_last]
        # 创建 DataFrame 并保存为 CSV
        eval_sas_labels_df = pd.DataFrame(
            np.array(sas_labels),
            index=[data_index[i] for i in eval_indices_last],  # 使用 coexistence_data.index 对应的索引
            columns=["Label"]
        )
        # 保存为 CSV 文件
        # eval_sas_labels_df.to_csv("./data_VAE/sas_eval_idx_labels.csv", index=True, header=True)
        eval_sas_labels_df.to_csv(self.save_data_label_path + "/sas_eval_idx_labels.csv", index=True, header=True)
        if self._config.with_cuda:
            eval_outputs = torch.cat(eval_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            eval_targets = torch.cat(eval_targets, dim=0).detach().cpu().numpy()
        else:
            eval_outputs = torch.cat(eval_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
            eval_targets = torch.cat(eval_targets, dim=0).detach().numpy()
        # self.cal_auc(eval_outputs, eval_targets, num_epochs)

        self.cal_auc(eval_outputs[-3864:], eval_targets[-3864:], "sas")

    def model_runner_single(self, virus_data, host_data, coexistence_data, target, to_valid_data_num=0, num_epochs=100):

        self.model.train()
        self.step = 0
        self.model = self.model.to(self.device)
        loss_total = []
        for epoch_ in range(num_epochs):
            epoch_res = []
            loss_epcoh = 0
            target_epoch = []

            for batchidx, (virus_data, host_data, target, idx_train) in enumerate(self.train_data):
                virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                    self.device)
                noise = self.make_noise(virus_data.shape[0], 10)

                # Forward pass
                output, decoded_outputs = self.model(virus_data, host_data, noise)
                virus_data = virus_data.to(dtype=torch.float32)
                loss = self.loss_fn(output, target) + 0.01 * self.recons_loss_fn(decoded_outputs[0], decoded_outputs[
                    1]) + self.VAE_KL_loss_fn(decoded_outputs[0], decoded_outputs[1])
                epoch_res.append(torch.sigmoid(output))
                target_epoch.append(target)
                loss.backward()
                self.step = batchidx
                self.optim.step()
                loss_epcoh = loss_epcoh + loss.item()

            loss_epcoh = loss_epcoh / len(self.train_data)
            print(f"Epoch [{epoch_ + 1}/{num_epochs}], Loss: {loss_epcoh:.4f}")
            loss_total.append(loss_epcoh / len(self.train_data))
            if (epoch_ + 1) % self._config.eval_freq == 0:
                if self.min_loss > loss_epcoh:
                    print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s" % (
                        epoch_, loss_epcoh, self.min_loss, self.save_model_path))
                    self.save(self.save_model_path)
                    self.min_loss = loss_epcoh

                with open(self.csv_path, 'a') as f:
                    f.write(f'{epoch_ + 1},{loss_epcoh:.6f}\n')
                if self._config.with_cuda:
                    outputs = torch.cat(epoch_res, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
                    targets = torch.cat(target_epoch, dim=0).detach().cpu().numpy()
                else:
                    outputs = torch.cat(epoch_res, dim=0).detach().numpy()  # 拼接并转为 numpy
                    targets = torch.cat(target_epoch, dim=0).detach().numpy()

                # eval_outputs = []
                # eval_targets = []
                # with torch.no_grad():
                #     for virus_data, host_data, target, _ in self.test_data:
                #         # for virus_data, host_data, target, _, simi_data in self.test_data:
                #         #     if torch.all(simi_data != 0):
                #         #         virus_data, host_data, target, simi_data = virus_data.to(self.device), host_data.to(
                #         #             self.device), target.to(self.device), simi_data.to(self.device)
                #         #     else:
                #         virus_data, host_data, target = virus_data.to(self.device), host_data.to(
                #             self.device), target.to(self.device)
                #         noise = self.make_noise(virus_data.shape[0], 10)
                #         # Forward pass
                #         output, decoded_output = self.model(virus_data, host_data, noise)
                #         # Collect outputs for AUC calculation
                #         eval_outputs.append(torch.sigmoid(output))
                #         eval_targets.append(target)
                #
                # if self._config.with_cuda:
                #     eval_outputs = torch.cat(eval_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
                #     eval_targets = torch.cat(eval_targets, dim=0).detach().cpu().numpy()
                # else:
                #     eval_outputs = torch.cat(eval_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
                #     eval_targets = torch.cat(eval_targets, dim=0).detach().numpy()
                # auroc, aupr, tss = self.cal_auc_epoch(eval_outputs, eval_targets, epoch_)
                # if self.max_tss < tss:
                #     print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s" % (
                #         epoch_, loss_epcoh, self.max_tss, self.save_tss_model_path))
                #     self.save(self.save_tss_model_path)
                #     self.max_tss = tss
                # count_greater_than_09 = (train_bst_result > 0.9821).sum().item()
                # print(count_greater_than_09)
        plt.plot(range(1, num_epochs + 1), loss_total, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(self.save_path + 'loss.png')
        plt.close()

        self.model.eval()

        self.model.load_state_dict(torch.load(self.save_model_path + '/pytorch_model.bin'))

        self.model.eval()
        train_val_res = []
        train_val_targets = []
        sampler_idx = []
        with torch.no_grad():
            for virus_data, host_data, target, idx_train in self.train_data:
                sampler_idx.append(idx_train)
                virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                    self.device)
                noise = self.make_noise(virus_data.shape[0], 10)
                # Forward pass
                output, decoded_output = self.model(virus_data, host_data, noise)
                # Collect outputs for AUC calculation
                train_val_res.append(torch.sigmoid(output))
                train_val_targets.append(target)

        coexistence_data_index = pd.read_csv("./data_VAE/association_vector.csv", index_col=0, header=None)
        # coexistence_data_index = pd.read_csv("./data_repeat/association_fold_vector.csv", index_col=0, header=None)

        sampler_idx = np.concatenate(sampler_idx)
        data_index = coexistence_data_index.index.tolist()
        coexistence_data_index = np.array(coexistence_data_index)
        train_labels = [coexistence_data_index[i] for i in sampler_idx]
        # 创建 DataFrame 并保存为 CSV
        last_train_labels_df = pd.DataFrame(
            np.array(train_labels),
            index=[data_index[i] for i in sampler_idx],  # 使用 coexistence_data.index 对应的索引
            columns=["Label"]
        )
        # 保存为 CSV 文件
        # last_train_labels_df.to_csv("./data_repeat/2025_06_24_sampler_train_last_idx_labels.csv", index=True, header=True)
        last_train_labels_df.to_csv(self.save_data_label_path + "/2025_06_24_sampler_train_last_idx_labels.csv", index=True,
                                    header=True)

        if self._config.with_cuda:
            train_val_res = torch.cat(train_val_res, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            train_val_targets = torch.cat(train_val_targets, dim=0).detach().cpu().numpy()
        else:
            train_val_res = torch.cat(train_val_res, dim=0).detach().numpy()  # 拼接并转为 numpy
            train_val_targets = torch.cat(train_val_targets, dim=0).detach().numpy()

        self.cal_auc_train_last(train_val_res, train_val_targets, num_epochs)

        test_outputs = []
        test_targets = []
        idx_tests = []

        with torch.no_grad():
            for virus_data, host_data, target, idx_test in self.test_data:
                virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                    self.device)

                idx_tests.append(idx_test)
                noise = self.make_noise(virus_data.shape[0], 10)
                # Forward pass
                output, decoded_output = self.model(virus_data, host_data, noise)
                # Collect outputs for AUC calculation
                test_outputs.append(torch.sigmoid(output))
                test_targets.append(target)

        if self._config.with_cuda:
            test_outputs = torch.cat(test_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            test_targets = torch.cat(test_targets, dim=0).detach().cpu().numpy()
        else:
            test_outputs = torch.cat(test_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
            test_targets = torch.cat(test_targets, dim=0).detach().numpy()
        self.cal_auc(test_outputs, test_targets, num_epochs)

        # coexistence_data_index = pd.read_csv("./data_repeat/association_fold_vector.csv", index_col=0, header=None)
        coexistence_data_index = pd.read_csv("./data_VAE/association_vector.csv", index_col=0, header=None)

        last_test_idx = np.concatenate(idx_tests)
        data_index = coexistence_data_index.index.tolist()
        coexistence_data_index = np.array(coexistence_data_index)
        test_labels = [coexistence_data_index[i] for i in last_test_idx]
        # 创建 DataFrame 并保存为 CSV
        test_labels_df = pd.DataFrame(
            np.array(test_labels),
            index=[data_index[i] for i in last_test_idx],  # 使用 coexistence_data.index 对应的索引
            columns=["Label"]
        )
        # 保存为 CSV 文件
        test_labels_df.to_csv(self.save_data_label_path + "/test_last_idx_labels.csv", index=True, header=True)
        # test_labels_df.to_csv("./data_VAE/test_last_idx_labels.csv", index=True, header=True)

        #val loader
        eval_outputs = []
        eval_targets = []
        eval_indices = []
        with (torch.no_grad()):
            for virus_data, host_data, target, eval_index in self.val_data:
                virus_data, host_data, target = virus_data.to(self.device), host_data.to(self.device), target.to(
                    self.device)

                noise = self.make_noise(virus_data.shape[0], 10)
                # Forward pass
                output, decoded_output = self.model(virus_data, host_data, noise)
                # Collect outputs for AUC calculation
                eval_outputs.append(torch.sigmoid(output))
                eval_targets.append(target)
                eval_indices.append(eval_index)

        coexistence_data_sas_index = pd.read_csv("./data_VAE/SARS_association_vector.csv", index_col=0, header=None)
        eval_indices = np.array(eval_index)
        data_index = coexistence_data_sas_index.index.tolist()
        coexistence_data_sas_index = np.array(coexistence_data_sas_index)
        sas_labels = [coexistence_data_sas_index[i] for i in eval_indices]
        # 创建 DataFrame 并保存为 CSV
        eval_sas_labels_df = pd.DataFrame(
            np.array(sas_labels),
            index=[data_index[i] for i in eval_indices],  # 使用 coexistence_data.index 对应的索引
            columns=["Label"]
        )
        # 保存为 CSV 文件
        # eval_sas_labels_df.to_csv("./data_VAE/sas_eval_idx_labels.csv", index=True, header=True)
        eval_sas_labels_df.to_csv(self.save_data_label_path + "/sas_eval_idx_labels.csv", index=True, header=True)
        if self._config.with_cuda:
            eval_outputs = torch.cat(eval_outputs, dim=0).detach().cpu().numpy()  # 拼接并转为 numpy
            eval_targets = torch.cat(eval_targets, dim=0).detach().cpu().numpy()
        else:
            eval_outputs = torch.cat(eval_outputs, dim=0).detach().numpy()  # 拼接并转为 numpy
            eval_targets = torch.cat(eval_targets, dim=0).detach().numpy()
        # self.cal_auc(eval_outputs, eval_targets, num_epochs)

        self.cal_auc(eval_outputs[-3864:], eval_targets[-3864:], "sas")
