import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
import os
import torch.nn.init as init


class VirusHostCoexistenceModel(nn.Module):
    def __init__(self, input_dim_virus, input_dim_host, hidden_dim, transformer_dim, output_dim, pool_size,
                 dropout_rate=0.5):
        super(VirusHostCoexistenceModel, self).__init__()

        # GAT Layers for Virus-Virus, Host-Host, and Virus-Host Graphs
        self.gat_virus = GATConv(input_dim_virus, hidden_dim, heads=3, concat=False)
        self.gat_host = GATConv(input_dim_host, hidden_dim, heads=3, concat=False)
        self.gat_virus_host = GATConv(input_dim_host, hidden_dim, heads=3, concat=False)
        self.gat_host_virus = GATConv(input_dim_virus, hidden_dim, heads=3, concat=False)

        self.gat_virus_linear = nn.Linear(input_dim_virus,hidden_dim)
        self.gat_host_linear = nn.Linear(input_dim_host,hidden_dim)
        self.gat_virus_host_linear = nn.Linear(input_dim_host,hidden_dim)
        self.gat_host_virus_linear = nn.Linear(input_dim_virus,hidden_dim)

        # GCN Layers
        self.gcn_virus = GCNConv(hidden_dim, hidden_dim)
        self.gcn_host = GCNConv(hidden_dim, hidden_dim)
        self.gcn_co_host = GCNConv(hidden_dim, hidden_dim)
        self.gcn_co_virus = GCNConv(hidden_dim, hidden_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Pooling layer to ensure consistent sizes
        self.pool = nn.AdaptiveAvgPool1d(pool_size)  # Pooling to specified pool_size

        # Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=4 * pool_size, nhead=8, dropout=dropout_rate)
        self.transformer_pool = nn.TransformerEncoder(encoder_layer, num_layers=2)

        encoder_layer_virus = nn.TransformerEncoderLayer(d_model=input_dim_host, nhead=2, dropout=dropout_rate)
        self.transformer_virus = nn.TransformerEncoder(encoder_layer_virus, num_layers=4)

        # MLP Layers for output prediction
        self.mlp_host = nn.Sequential(
            nn.Linear(4 * pool_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim_host)
        )
        self.mlp_virus = nn.Sequential(
            nn.Linear(4 * pool_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim_virus),
            # nn.Softmax()
        )
        self.sof = nn.Softmax()
        self.mlp_out_host = nn.Sequential(
            nn.Linear(input_dim_virus * 3, input_dim_virus * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim_virus * 2, input_dim_virus),
        )
        self.mlp_out_virus = nn.Sequential(
            nn.Linear(input_dim_host * 3, input_dim_host * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim_host * 2, input_dim_host),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(pool_size, input_dim_host),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(input_dim_host * 3, input_dim_host * 1),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(input_dim_host * 3, input_dim_host),
        )
        self.last_out = nn.Sequential(
            nn.Linear(input_dim_host * 3, input_dim_host * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim_host * 2, input_dim_host * 1),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(input_dim_host * 3, input_dim_host),
        )
        self.norm = nn.BatchNorm1d(128)
        self.rel = nn.LeakyReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Apply Xavier initialization to Linear layers
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Apply Xavier initialization to Conv2d layers (if you have any)
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Apply initialization to BatchNorm layers (if you have any)
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, GATConv):
                # GATConv might not have standard weight initialization; you may need to initialize manually
                if hasattr(m, 'weight'):
                    init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, virus_data, host_data, coexistence_data, virus_edge_index, host_edge_index,
                coexistence_edge_index, coexistence_edge_index_t, virus_edge_weight, host_edge_weight):
        # GAT Outputs
        virus_features, virus_attention_weights = self.gat_virus(virus_data, virus_edge_index,
                                                                 edge_attr=virus_edge_weight,
                                                                 return_attention_weights=True)
        host_features, host_attention_weights = self.gat_host(host_data, host_edge_index, edge_attr=host_edge_weight,
                                                              return_attention_weights=True)
        coexistence_features, VH_virus_attention_weights = self.gat_virus_host(coexistence_data,
                                                                               coexistence_edge_index_t,
                                                                               return_attention_weights=True)
        coexistence_features_host, VH_host_attention_weights = self.gat_host_virus(coexistence_data.t(),
                                                                                   coexistence_edge_index,
                                                                                   return_attention_weights=True)
        #
        # output_host = torch.mm(virus_features, host_features.t())
        # output_virus = torch.mm(coexistence_features,coexistence_features_host.t())
        # output = output_host, output_virus, output_host + output_virus, virus_attention_weights, host_attention_weights, VH_virus_attention_weights, VH_host_attention_weights
        # return output

        virus_hidden = self.gat_virus_linear(virus_data)
        virus_hidden = self.norm(virus_hidden)
        virus_hidden = self.rel(virus_hidden)
        host_hidden = self.gat_host_linear(host_data)
        host_hidden = self.norm(host_hidden)
        host_hidden = self.rel(host_hidden)
        # coexistence_hidden_host = self.gat_virus_host_linear(coexistence_data)
        # coexistence_hidden_virus = self.gat_host_virus_linear(coexistence_data.t())
        output_host = torch.mm(virus_hidden, host_hidden.t())
        output_virus = torch.mm(host_hidden, virus_hidden.t()).t()

        # output_virus = torch.mm(coexistence_features,coexistence_features_host.t())
        output = output_host, output_virus, output_host + output_virus, virus_attention_weights, host_attention_weights, VH_virus_attention_weights, VH_host_attention_weights
        return output
        # Apply Dropout
        virus_hidden = self.dropout(virus_features)
        host_hidden = self.dropout(host_features)
        coexistence_hidden_host = self.dropout(coexistence_features)
        coexistence_hidden_virus = self.dropout(coexistence_features_host)

        # GCN Outputs
        virus_hidden = self.gcn_virus(virus_hidden, virus_edge_index)
        host_hidden = self.gcn_host(host_hidden, host_edge_index)
        coexistence_hidden_host = self.gcn_co_host(coexistence_hidden_host, coexistence_edge_index_t)
        coexistence_hidden_virus = self.gcn_co_virus(coexistence_hidden_virus, coexistence_edge_index)
        # coexistence_hidden = self.gcn_co_host(coexistence_features, coexistence_edge_index)

        # Apply Dropout
        virus_hidden = self.dropout(virus_hidden)
        host_hidden = self.dropout(host_hidden)
        coexistence_hidden_host = self.dropout(coexistence_hidden_host)
        coexistence_hidden_virus = self.dropout(coexistence_hidden_virus)
        # coexistence_hidden = self.dropout(coexistence_hidden)

        # # Pooling to reduce size and match dimensions
        # virus_hidden_pooled = self.pool(virus_hidden.unsqueeze(0)).squeeze(0)  # Shape: [pool_size, hidden_dim]
        # host_hidden_pooled = self.pool(host_hidden.unsqueeze(0)).squeeze(0)  # Shape: [pool_size, hidden_dim]
        coexistence_hidden_host_pooled = self.pool(coexistence_hidden_host.unsqueeze(0)).squeeze(0)
        coexistence_hidden_virus_pooled = self.pool(coexistence_hidden_virus.unsqueeze(0)).squeeze(0)
        # coexistence_hidden_pooled = self.pool(coexistence_hidden.unsqueeze(0)).squeeze(0)
        virus_hidden_pooled = self.pool(virus_hidden.transpose(0, 1)).transpose(0, 1)  # Shape: [128, hidden_dim]
        host_hidden_pooled = self.pool(host_hidden.transpose(0, 1)).transpose(0, 1)  # Shape: [128, hidden_dim]
        coexistence_hidden_host_pooled_trans = self.pool(coexistence_hidden_host.transpose(0, 1)).transpose(0, 1)
        coexistence_hidden_virus_pooled_trans = self.pool(coexistence_hidden_virus.transpose(0, 1)).transpose(0, 1)
        # coexistence_hidden_virus_pooled = self.pool(coexistence_hidden.transpose(0, 1)).transpose(0,1)

        # Concatenate pooled hidden vectors
        combined_features = torch.cat(
            (virus_hidden_pooled, host_hidden_pooled, coexistence_hidden_host_pooled_trans,
             coexistence_hidden_virus_pooled_trans),
            dim=-1)
        # combined_features = torch.cat(
        #     (virus_hidden_pooled, host_hidden_pooled, coexistence_hidden_host_pooled_trans+coexistence_hidden_virus_pooled_trans),
        #     dim=-1)
        # combined_features = torch.cat(
        #     (virus_hidden_pooled, host_hidden_pooled),dim=-1)
        # Transformer Layer
        transformer_output = self.transformer_pool(combined_features.unsqueeze(0)).squeeze(0)

        # MLP Output
        output_host = self.mlp_host(transformer_output)
        output_virus = self.mlp_virus(transformer_output)

        output_host = torch.mul(output_host, host_hidden.t())
        output_virus = torch.mul(output_virus, virus_hidden.t())
        output_matrix = torch.mm(output_host.T, output_virus)
        hidden_output = torch.mm(host_hidden, virus_hidden.t())
        coexistence_matrix = torch.mm(coexistence_hidden_virus_pooled, coexistence_hidden_host_pooled.t())
        output_matrix_host = torch.cat((output_matrix, hidden_output, coexistence_matrix), dim=-1)
        output_matrix_virus = torch.cat((output_matrix, hidden_output, coexistence_matrix), dim=0).t()
        # output_matrix_virus = torch.cat((output_matrix, hidden_output), dim=0).t()
        # output_matrix_host = torch.cat((output_matrix, hidden_output), dim=-1)

        # output_matrix = output_matrix + coexistence_matrix
        output_host = self.mlp_out_host(output_matrix_host)
        output_virus = self.mlp_out_virus(output_matrix_virus)
        # output_virus_hidden = torch.mm(hidden_output.t(), output_host)# * output_virus.mean(-1)
        # transformer_output_virus1 = self.transformer_virus(output_virus_hidden.unsqueeze(0)).squeeze(0)
        # transformer_output_virus2 = self.pool(output_virus_hidden.unsqueeze(0)).squeeze(0)
        # # transformer_output_virus = transformer_output_virus2+transformer_output_virus1
        # output_virus = self.mlp_out(transformer_output_virus2)
        # output_virus = output_virus + transformer_output_virus1
        # transformer_output_virus1 = self.transformer_virus(output_host.t().unsqueeze(0)).squeeze(0)
        # transformer_output_virus2 = self.transformer_virus(output_virus.unsqueeze(0)).squeeze(0)
        # transformer_output_virus1 = self.last_out(transformer_output_virus1)
        # co_output_virus = torch.cat((output_host.t(), output_virus, output_host.t() + output_virus), dim=-1)
        # transformer_output_virus2 = self.last_out(co_output_virus)

        output = output_host.t(), output_virus, output_host.t() + output_virus, virus_attention_weights, host_attention_weights, VH_virus_attention_weights, VH_host_attention_weights
        return output

    def save_pretrained(self, file_path):
        # 创建目录（如果不存在）
        os.makedirs(file_path, exist_ok=True)

        # 保存模型权重
        model_weights_path = os.path.join(file_path, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_weights_path)


class SimplifiedVirusHostModel(nn.Module):
    def __init__(self, input_dim_virus, input_dim_host, hidden_dim, output_dim, pool_size, dropout_rate=0.5):
        super(SimplifiedVirusHostModel, self).__init__()

        # GAT Layers
        # self.gat_virus = GATConv(input_dim_virus, hidden_dim, heads=3, concat=False)
        # self.gat_host = GATConv(input_dim_host, hidden_dim, heads=3, concat=False)
        # self.gat_virus_host = GATConv(input_dim_host, hidden_dim, heads=3, concat=False)

        self.gat_virus = nn.Linear(input_dim_virus,hidden_dim)
        self.gat_host = nn.Linear(input_dim_host,hidden_dim)
        self.gat_virus_host = nn.Linear(input_dim_host,hidden_dim)
        self.gat_host_virus = GATConv(input_dim_virus, hidden_dim, heads=3, concat=False)

        # GCN Layers
        self.gcn_virus = GCNConv(hidden_dim, hidden_dim)
        self.gcn_host = GCNConv(hidden_dim, hidden_dim)
        self.gcn_co_host = GCNConv(hidden_dim, hidden_dim)
        self.gcn_co_virus = GCNConv(hidden_dim, hidden_dim)



        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(pool_size)

        # Transformer Layer
        transformer_dim = hidden_dim * 2  # Adjusted d_model for Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output MLP
        self.mlp_output = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GATConv) or isinstance(m, GCNConv):
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, virus_data, host_data, coexistence_data, virus_edge_index, host_edge_index,
                coexistence_edge_index):
        # GAT Outputs
        virus_features = self.gat_virus(virus_data, virus_edge_index)
        host_features = self.gat_host(host_data, host_edge_index)
        coexistence_features = self.gat_virus_host(coexistence_data, coexistence_edge_index)

        # Apply Dropout
        virus_hidden = self.dropout(virus_features)
        host_hidden = self.dropout(host_features)
        coexistence_hidden = self.dropout(coexistence_features)

        # GCN Outputs
        virus_hidden = self.gcn_virus(virus_hidden, virus_edge_index)
        host_hidden = self.gcn_host(host_hidden, host_edge_index)

        # Pooling and Concatenation
        virus_hidden_pooled = self.pool(virus_hidden.transpose(0, 1)).transpose(0, 1)
        host_hidden_pooled = self.pool(host_hidden.transpose(0, 1)).transpose(0, 1)
        combined_features = torch.cat((virus_hidden_pooled, host_hidden_pooled), dim=-1)

        # Transformer Layer
        transformer_output = self.transformer(combined_features.unsqueeze(0)).squeeze(0)

        # MLP Output
        output = self.mlp_output(transformer_output)
        return output