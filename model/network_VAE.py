import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
import os
import torch.nn.init as init


class LearnablePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(LearnablePooling, self).__init__()
        self.attn = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float32))  # Learnable parameters

    def forward(self, x):
        attn_weights = torch.matmul(x, self.attn)  # Calculate attention over sequence
        attn_weights = torch.softmax(attn_weights, dim=1)
        return torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)


class VAE_transmission_model(nn.Module):
    def __init__(self, input_dim_virus, input_dim_host, hidden_dim, max_frequence, embed_dim, dropout_rate=0.5):
        super(VAE_transmission_model, self).__init__()

        self.embedding = nn.Embedding(max_frequence, embed_dim)  # 4个字母，嵌入到embed_dim维
        # 1D 卷积层，将输入的 256 个通道转换为 512 个通道
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        # 另一个 1D 卷积层，将 512 个通道转换为 1024 个通道
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        # 全局平均池化，将序列长度维度压缩掉
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # hiddem dim
        # self.virus_dim = [hidden_dim * 2, hidden_dim, 64]
        # self.virus_dim = [embed_dim, 2048, 1024, 512, 256, hidden_dim, 64]
        self.virus_dim = [embed_dim, hidden_dim, 64]

        self.virus_hamming_dim = [464, hidden_dim, 64]

        # self.host_dim = [input_dim_host, 2048, 1024, 512, 256, hidden_dim, 64]
        # self.host_dim = [input_dim_host, 512,256, hidden_dim, 64]
        self.host_dim = [input_dim_host, hidden_dim, 64]

        self.decoder_dim = [64 * 2 + 10, 256, 512]

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Virus encoder
        self.virus_encoder = self._build_encoder(self.virus_dim)
        # self.virus_hamming_encoder = self._build_encoder(self.virus_hamming_dim)

        # Host encoder
        self.host_encoder = self._build_encoder(self.host_dim)

        # Decoder
        self.decoder = self._build_decoder(self.decoder_dim)

        # Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=8, dropout=dropout_rate)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=dropout_rate)

        # encoder_layer_mapping = nn.TransformerEncoderLayer(d_model=64, nhead=8, dropout=dropout_rate)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # self.transformer_mapping = nn.TransformerEncoder(encoder_layer_mapping, num_layers=4)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.learnabel_pool = LearnablePooling(hidden_dim * 2)
        self.learnabel_pool = LearnablePooling(embed_dim)

        # classifier MLP
        self.classifier = nn.Sequential(
            # nn.Linear(2048, 1024),  # 将 feature_dim 映射到更小的特征空间
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(1025, 512),  # 将 feature_dim 映射到更小的特征空间
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(512, 64),  # 将 feature_dim 映射到更小的特征空间
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),  # 映射到类别数

        )
        # self.nor = nn.BatchNorm1d(embed_dim)
        # self._initialize_weights()

    def _build_encoder(self, dims):
        """
        Build encoder network based on provided dimensions.
        Each encoder consists of Linear layers with ReLU activation and Dropout.
        """
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(self.dropout)  # Apply dropout for regularization
        return nn.Sequential(*layers)

    def _build_decoder(self, dims):
        """
        Build decoder network based on provided dimensions.
        Each layer has Linear transformations followed by ReLU.
        """
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, GATConv) or isinstance(m, GCNConv):
    #             if hasattr(m, 'weight'):
    #                 nn.init.xavier_uniform_(m.weight)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, virus_data, host_data, noise=None):
        # Pass through virus encoder
        # x = self.transformer_mapping(virus_data)
        # virus_data_raw = virus_data
        x = self.embedding(virus_data)

        # virus_data = x.mean(1)

        # x, _ = self.lstm(x)  # x.shape = (batch_size, seq_len, hidden_dim * 2)
        # Optionally, use only the final output of the sequence
        virus_data_mean = x.mean(1)
        # virus_data_mean = self.nor(virus_data_mean)
        # virus_data_mean = self.learnabel_pool(x)
        # virus_data = x[:, -1, :]  # [batch_size, hidden_dim * 2]
        # 将数据转换为 (batch_size, channels, sequence_length) 以适应 Conv1d
        # x = x.permute(0, 2, 1)  # (batch_size, 1024, 256)
        #
        # # 第一个卷积块
        # x = self.conv1(x)  # (batch_size, 512, 256)
        # x = self.bn1(x)
        # x = self.relu1(x)
        #
        # # 第二个卷积块
        # x = self.conv2(x)  # (batch_size, 1024, 256)
        # x = self.bn2(x)
        # x = self.relu2(x)
        #
        # # 全局平均池化，将序列长度维度压缩为 1
        # x = self.global_avg_pool(x)  # (batch_size, 1024, 1)
        #
        # # 去除多余的维度，得到最终形状 (batch_size, 1024)
        # x = x.squeeze(-1)  # (batch_size, 1024)

        # x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim) for Transformer
        # virus_data = self.transformer(virus_data)  # (seq_len, batch_size, embed_dim)
        # x = x.permute(1, 2, 0)  # (batch_size, embed_dim, seq_len)
        # x = self.pooling(x)
        # x = x.squeeze(-1)       # (batch_size, embed_dim)

        virus_encoded = self.virus_encoder(virus_data_mean)
        # virus_hamming_encoded = self.virus_hamming_encoder(virus_data)

        # Pass through host encoder
        host_encoded = self.host_encoder(host_data)

        # Concatenate encoded features
        # if noise is None:
        # combined_features = torch.cat((virus_encoded, host_encoded), dim=1)
        # else:
        combined_features = torch.cat((virus_encoded, host_encoded,noise), dim=1)
        # combined_features = torch.cat((host_encoded, virus_encoded, noise), dim=1)

        # Pass through decoder
        decoded_output = self.decoder(combined_features)
        # decoded_output = decoded_output  #+virus_data_mean
        # decoded_output = torch.cat((decoded_output, virus_data_mean), dim=1)

        # transformed_output = self.transformer(decoded_output)
        # out = self.classifier(combined_features)
        out = self.classifier(decoded_output)

        return out, decoded_output

    def save_pretrained(self, file_path):
        # 创建目录（如果不存在）
        os.makedirs(file_path, exist_ok=True)

        # 保存模型权重
        model_weights_path = os.path.join(file_path, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_weights_path)
