import torch
import torch.nn as nn
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Learnable Pooling Module
class LearnablePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(LearnablePooling, self).__init__()
        self.attn = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float32))  # Learnable parameters

    def forward(self, x):
        attn_weights = torch.matmul(x, self.attn)  # Calculate attention over sequence
        attn_weights = torch.softmax(attn_weights, dim=1)
        return torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)


# Encoder Module
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = self._build_encoder(input_dim, hidden_dims)

    def _build_encoder(self, input_dim, dims):
        layers = []
        # layers.append(nn.Embedding(input_dim, dims[0]))  # Embedding layer
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(self.dropout)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# Decoder Module
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = self._build_decoder(input_dim, hidden_dims)

    def _build_decoder(self, input_dim, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(self.dropout)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNFeatureExtractor, self).__init__()
        # Define a simple CNN architecture
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)  # Batch normalization after conv1
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # Max pooling layer after conv1, adjusted stride

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)  # Batch normalization after conv2
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # Max pooling layer after conv2, adjusted stride

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)  # Batch normalization after conv3
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # Max pooling layer after conv3, adjusted stride

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(256, output_dim)  # Fully connected layer to reduce features

    def forward(self, x):
        # x = x.squeeze(-1)
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)  # Batch normalization
        x = torch.relu(x)
        x = self.pool1(x)  # Max pooling

        x = self.conv2(x)
        x = self.bn2(x)  # Batch normalization
        x = torch.relu(x)
        x = self.pool2(x)  # Max pooling

        x = self.conv3(x)
        x = self.bn3(x)  # Batch normalization
        x = torch.relu(x)
        x = self.pool3(x)  # Max pooling

        # Global average pooling to condense the features
        x = self.global_pool(x.permute(0, 2, 1))
        x = x.squeeze(-1)  # Remove the sequence length dimension

        # Fully connected layer
        # x = self.fc(x)
        return x


class CNNFeatureExtractor3D(nn.Module):
    def __init__(self,
                 embed_dim: int,  # 嵌入维度
                 seq_length: int,  # 原始序列长度
                 num_channels: int = 64,  # 初始通道数
                 output_dim: int = 256):  # 最终输出维度
        super().__init__()

        # 输入形状: (batch_size, embed_dim, seq_length)
        self.conv_layers = nn.Sequential(
            # 第一卷积块
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels * 2, kernel_size=5, padding=1),
            nn.BatchNorm1d(num_channels * 2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),  # 序列长度变为 ceil(seq_length/2)

            # 第二卷积块
            nn.Conv1d(num_channels * 2, num_channels * 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels * 1),
            nn.GELU(),
            nn.MaxPool1d(2),  # 序列长度变为 ceil(seq_length/4)

            # # 第三卷积块 (空洞卷积扩大感受野)
            # nn.Conv1d(num_channels * 3, num_channels *2, kernel_size=3,
            #           dilation=2, padding=2),
            # nn.BatchNorm1d(num_channels * 2),
            # nn.GELU(),
            # nn.AdaptiveMaxPool1d(output_size=16),  # 固定输出长度
            #
            # # 第四卷积块
            # nn.Conv1d(num_channels * 2, num_channels * 1, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_channels * 1),
            # nn.GELU()
        )

        # 动态计算全连接层输入维度
        self._initialize_parameters(seq_length)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def _initialize_parameters(self, seq_length):
        # 通过模拟前向传播计算最终维度
        test_input = torch.randn(2, self.conv_layers[0].in_channels, seq_length)
        with torch.no_grad():
            test_output = self.conv_layers(test_input)
        self.fc_input_dim = test_output.size(1) * test_output.size(2)

    def forward(self, x):
        """
        输入形状: (batch_size, seq_length, embed_dim)
        输出形状: (batch_size, output_dim)
        """
        # 调整维度顺序为 (batch, channels, sequence)
        x = x.permute(0, 2, 1)  # [B, E, S]

        # 通过卷积层
        conv_features = self.conv_layers(x)  # [B, C, S']

        # 展平特征
        flattened = conv_features.view(conv_features.size(0), -1)  # [B, C*S']

        # 全连接层
        return self.fc(flattened)


class Custom1DCNN_virus(nn.Module):
    def __init__(self, output_dim=512):
        super(Custom1DCNN_virus, self).__init__()

        # 输入形状调整：[b, 256] -> [b, 1, 256]（添加通道维度）
        self.network = nn.Sequential(
            # 第1卷积层
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 输出长度：256/2=128

            # 第2卷积层
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出长度：128/2=64

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出长度：64/2=32


            # 全局平均池化（压缩空间维度）
            # nn.AdaptiveAvgPool1d(1),  # 输出形状：[b, 32, 1]

            # 全连接层调整维度
            nn.Flatten(),
            nn.Linear(3712, output_dim)
        )

    def forward(self, x):
        # 原始输入形状：[b, 256]
        x = x.unsqueeze(1)  # 添加通道维度 -> [b, 1, 256]
        return self.network(x)

class Custom1DCNN_host(nn.Module):
    def __init__(self, output_dim=512):
        super(Custom1DCNN_host, self).__init__()

        # 输入形状调整：[b, 256] -> [b, 1, 256]（添加通道维度）
        self.network = nn.Sequential(
            # 第1卷积层
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 输出长度：256/2=128

            # 第2卷积层
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出长度：128/2=64

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 输出长度：64/2=32


            # 全局平均池化（压缩空间维度）
            # nn.AdaptiveAvgPool1d(1),  # 输出形状：[b, 32, 1]

            # 全连接层调整维度
            nn.Flatten(),
            nn.Linear(5120, output_dim)
        )

    def forward(self, x):
        # 原始输入形状：[b, 256]
        x = x.unsqueeze(1)  # 添加通道维度 -> [b, 1, 256]
        return self.network(x)


class EnhancedVAE(nn.Module):
    def __init__(self,
                 input_dim_virus: int,
                 input_dim_host: int,
                 max_frequence: int,
                 embed_dim: int = 1024,
                 seq_length: int = 464,  # 病毒序列原始长度
                 dropout_rate: float = 0.2):
        super().__init__()
        self.platt_A = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
        self.platt_B = nn.Parameter(torch.tensor(0.0))  # 初始化为0.0
        # 病毒特征处理
        # self.virus_embedding = nn.Embedding(max_frequence, embed_dim)
        self.cnn_feature_extractor = CNNFeatureExtractor3D(
            embed_dim=embed_dim,
            seq_length=seq_length,
            output_dim=256
        )
        # 宿主特征处理
        self.virus_encoder = nn.Sequential(
            nn.Linear(input_dim_virus, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        # 宿主特征处理
        self.host_encoder = nn.Sequential(
            nn.Linear(input_dim_host, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(266, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(256, 512),
            # nn.LayerNorm(512),
            # nn.GELU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(512, 512 * 2),

        )
        # 联合特征处理
        self.joint_processor = nn.Sequential(
            #256 + 256 + 10 + 464
            nn.Linear(512, 512),  # 病毒CNN特征 + 宿主特征
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        self.latent_feature_extractor = nn.Sequential(
            nn.Unflatten(1, (1, 16, 32)),  # 将976维向量转为1x16x61的"图像"
            nn.Conv2d(1, 12, kernel_size=3, stride=2),  # 输出[16, 7, 30]
            nn.GELU(),
            nn.MaxPool2d(2),  # 输出[16, 3, 15]
            nn.Flatten(),  # 展平为16*3*15=720
            nn.Linear(252, 128)  # 降维到256
        )

        # 分类器
        self.classifier = nn.Sequential(
            # nn.Linear(976, 256),
            # nn.LayerNorm(128),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 32),
            # nn.LayerNorm(32),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        self.mlp_virus_cnn = Custom1DCNN_virus()
        self.mlp_host_cnn = Custom1DCNN_host()

        self.apply(self._init_weights)
        self.fc_mu = nn.Linear(1024, 266)
        self.fc_log_var = nn.Linear(1024, 266)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, virus_data, host_data, noise=None):
        # virus_data_norm = 1 - (virus_data - torch.min(virus_data)) / (torch.max(virus_data) - torch.min(virus_data))
        # virus_features_norm = self.mlp_virus_cnn(virus_data)

        # 病毒特征处理
        # virus_embed = self.virus_embedding(virus_data)  # [B, S, E]
        # virus_features = self.cnn_feature_extractor(virus_embed)  # [B, 256]
        virus_features_norm = self.virus_encoder(virus_data)  # [B, 256]
        # virus_data_norm_np = virus_data_norm.cpu().detach().numpy().flatten()  # 转换为numpy并展平
        #
        # virus_features_np = virus_features.cpu().detach().numpy().flatten()  # 转换为numpy并展平
        # virus_features_norm_np = virus_features_norm.cpu().detach().numpy().flatten()  # 转换为numpy并展平
        #
        # plt.figure(figsize=(12, 6))
        #
        # # 绘制归一化数据的密度图
        # # sns.kdeplot(virus_data_norm_np, fill=True, color="blue", alpha=0.3, label="Normalized Input Data")
        #
        # # 绘制特征提取后的密度图
        # sns.kdeplot(virus_features_np, fill=True, color="red", alpha=0.3, label="CNN Features")
        # sns.kdeplot(virus_features_norm_np, fill=True, color="green", alpha=0.3, label="virus_features_norm Features")
        #
        # # 设置标题和标签
        # plt.title("Density Comparison: Normalized Input vs. CNN Features", fontsize=14)
        # plt.xlabel("Value", fontsize=12)
        # plt.ylabel("Density", fontsize=12)
        # plt.legend(loc="upper right")  # 显示图例
        #
        # plt.show()
        # # 宿主特征处理
        # host_features = self.mlp_host_cnn(host_data)

        host_features = self.host_encoder(host_data)  # [B, 256]

        # 特征融合
        combined = torch.cat([virus_features_norm, host_features], dim=1)  # [B, 512]

        joint_features = self.joint_processor(combined)  # [B, 256]
        #
        combined_z = torch.cat([joint_features, noise], dim=1)  # [B, 512]

        # mu = self.fc_mu(joint_features)  # [B, latent_dim]
        # log_var = self.fc_log_var(joint_features)  # [B, latent_dim]

        # 重参数化采样
        # z = self.reparameterize(mu, log_var)  # [B, latent_dim]

        decoded_latent = self.decoder(combined_z)
        # mu = self.fc_mu(decoded_latent)  # [B, latent_dim]
        # log_var = self.fc_log_var(decoded_latent)  # [B, latent_dim]
        # decoded_output = self.reparameterize(mu, log_var)  # [B, latent_dim]
        mu = decoded_latent[:, :512]
        log_var = decoded_latent[:, 512:]
        decoded_output = self.reparameterize(mu, log_var)
        # 分类输出
        extracted_decoded_output = self.latent_feature_extractor(decoded_output)
        logits = self.classifier(extracted_decoded_output)  # [B, 2]
        # calibrated_probs = torch.sigmoid(self.platt_A * logits + self.platt_B)
        decoded_output_para = [mu, log_var]
        decoded_outputs = [combined, decoded_output, decoded_output_para]
        return logits, decoded_outputs#,host_features,virus_features_norm,extracted_decoded_output

    def save_pretrained(self, file_path, step):
        os.makedirs(file_path, exist_ok=True)
        model_weights_path = os.path.join(file_path, f'pytorch_model{step}.bin')
        torch.save(self.state_dict(), model_weights_path)


# VAE Transmission Model
class VAE_transmission_model(nn.Module):
    def __init__(self, input_dim_virus, input_dim_host, hidden_dim, max_frequence, embed_dim, dropout_rate=0.2):
        super(VAE_transmission_model, self).__init__()

        self.embedding = nn.Embedding(max_frequence, embed_dim)
        # Encoder & Decoder Setup
        self.virus_encoder = Encoder(input_dim_virus, [embed_dim, 512, 256, hidden_dim, 64], dropout_rate)
        self.host_encoder = Encoder(input_dim_host, [input_dim_host, 512, 256, hidden_dim, 64], dropout_rate)
        self.decoder = Decoder(64 * 2 + 10, [64 * 2 + 10, 256, 512, 1024], dropout_rate)

        # CNN Feature Extractor
        self.cnn_feature_extractor = CNNFeatureExtractor(input_dim=1, output_dim=64)
        # self.cnn_feature_extractor = CNNFeatureExtractor3D(
        #     embed_dim=embed_dim,
        #     seq_length=64,
        #     output_dim=256
        # )
        # Learnable Pooling
        self.learnable_pool = LearnablePooling(hidden_dim * 2)

        # Classifier MLP
        # 在分类器中添加更多正则化
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),  # 降低dropout率
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, virus_data, host_data, noise=None):

        virus_data_embedding = self.embedding(virus_data)

        virus_data = virus_data_embedding.mean(1)
        # Pass through virus encoder
        virus_encoded = self.virus_encoder(virus_data)

        # Pass through host encoder
        host_encoded = self.host_encoder(host_data)

        # Concatenate encoded features
        if noise is not None:
            combined_features = torch.cat((virus_encoded, host_encoded, noise), dim=1)
        else:
            combined_features = torch.cat((virus_encoded, host_encoded), dim=1)

        # Pass through decoder
        decoded_output = self.decoder(combined_features)

        # Concatenate the decoded output and the virus data_2025_07_22 mean (additional feature)
        combined_features_with_mean = torch.cat((decoded_output, virus_data), dim=1)

        # Apply CNN feature extraction to the concatenated features
        cnn_features = self.cnn_feature_extractor(combined_features_with_mean.unsqueeze(1))  # Add channel dimension

        # Classifier Output
        out = self.classifier(cnn_features)

        return out, combined_features

    def save_pretrained(self, file_path, step):
        os.makedirs(file_path, exist_ok=True)
        model_weights_path = os.path.join(file_path, f'pytorch_model{step}.bin')
        torch.save(self.state_dict(), model_weights_path)
