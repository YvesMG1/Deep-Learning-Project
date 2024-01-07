
import torch
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
import torch.nn.functional as F


class GATBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout_rate=0.5, batch_norm=True, residual=False):
        super(GATBlock, self).__init__()

        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout_rate)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_norm = batch_norm
        self.residual = residual

        if self.residual:
            self.res_connection = torch.nn.Linear(in_channels, out_channels * heads, bias=False)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels * heads)

    def forward(self, x, edge_index):
        res = x
        x = self.conv(x, edge_index)
        if self.residual:
            res = self.res_connection(res)
            x = x + res[:x.size(0)]
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, batch_norm=True, residual=False):
        super(GCNBlock, self).__init__()

        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_norm = batch_norm
        self.residual = residual

        if self.residual:
            self.res_connection = torch.nn.Linear(in_channels, out_channels, bias=False)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        res = x
        x = self.conv(x, edge_index)
        if self.residual:
            x += self.res_connection(res)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
    
class SAGEBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, batch_norm=True, residual=True):
        super(SAGEBlock, self).__init__()

        self.conv = SAGEConv(in_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_norm = batch_norm
        self.residual = residual

        if self.residual:
            self.res_connection = torch.nn.Linear(in_channels, out_channels, bias=False)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        res = x
        x = self.conv(x, edge_index)
        if self.residual:
            x += self.res_connection(res)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ResGCN(torch.nn.Module):
    def __init__(self, num_features, layer_configs, mlp_config, num_classes):
        super(ResGCN, self).__init__()

        initial_layer = layer_configs[0]
        self.initial = GCNBlock(num_features, initial_layer['out_channels'], initial_layer['dropout_rate'], initial_layer['batch_norm'])

        self.hidden_layers = torch.nn.ModuleList()
        for layer_config in layer_configs[1:]:
            self.hidden_layers.append(GCNBlock(layer_config['in_channels'], layer_config['out_channels'], layer_config['dropout_rate'], layer_config['batch_norm'], residual=True))

        # Configurable MLP
        mlp_layers = []
        prev_channels = layer_configs[-1]['out_channels']
        for layer in mlp_config:
            mlp_layers.append(torch.nn.Linear(prev_channels, layer['out_channels']))
            if layer.get('batch_norm', False):
                mlp_layers.append(torch.nn.BatchNorm1d(layer['out_channels']))
            if layer.get('relu', True):
                mlp_layers.append(torch.nn.ReLU())
            if 'dropout_rate' in layer:
                mlp_layers.append(torch.nn.Dropout(layer['dropout_rate']))
            prev_channels = layer['out_channels']

        mlp_layers.append(torch.nn.Linear(prev_channels, num_classes))
        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index, batch):
        x = self.initial(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        return x
    
class ResGAT(torch.nn.Module):
    def __init__(self, num_features, layer_configs, mlp_config, num_classes):
        super(ResGAT, self).__init__()

        # GAT layers
        initial_layer = layer_configs[0]
        self.initial = GATBlock(num_features, initial_layer['out_channels'], initial_layer.get('heads', 1), initial_layer['dropout_rate'], initial_layer['batch_norm'])

        self.hidden_layers = torch.nn.ModuleList()
        for layer_config in layer_configs[1:]:
            self.hidden_layers.append(GATBlock(layer_config['in_channels'], layer_config['out_channels'], layer_config.get('heads', 1), layer_config['dropout_rate'], layer_config['batch_norm'], residual=True))

        # Configurable MLP
        mlp_layers = []
        prev_channels = layer_configs[-1]['out_channels'] * layer_configs[-1].get('heads', 1)
        for layer in mlp_config:
            mlp_layers.append(torch.nn.Linear(prev_channels, layer['out_channels']))
            if layer.get('batch_norm', False):
                mlp_layers.append(torch.nn.BatchNorm1d(layer['out_channels']))
            if layer.get('relu', True):
                mlp_layers.append(torch.nn.ReLU())
            if 'dropout_rate' in layer:
                mlp_layers.append(torch.nn.Dropout(layer['dropout_rate']))
            prev_channels = layer['out_channels']

        mlp_layers.append(torch.nn.Linear(prev_channels, num_classes))
        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index, batch):
        x = self.initial(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        return x



class SAGE_LSTM(torch.nn.Module):
    def __init__(self, num_features, layer_configs, num_classes):
        super(SAGE_LSTM, self).__init__()

        initial_layer = layer_configs[0]
        self.initial = SAGEBlock(num_features, initial_layer['out_channels'], initial_layer['dropout_rate'], initial_layer['batch_norm'])

        self.hidden_layers = torch.nn.ModuleList()
        for layer_config in layer_configs[1:]:
            self.hidden_layers.append(SAGEBlock(layer_config['in_channels'], layer_config['out_channels'], layer_config['dropout_rate'], layer_config['batch_norm']))

        self.lstm_pooling = LSTM(input_size=layer_configs[-1]['out_channels'], hidden_size=128, batch_first=True)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.initial(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        # Process each graph in the batch individually and use LSTM on its nodes
        pooled_outputs = []
        for graph_id in batch.unique():
            # Extract nodes for this graph
            nodes_for_graph = x[batch == graph_id]

            # Add an extra dimension for batch (LSTM expects 3D input: batch x seq x feature)
            nodes_for_graph = nodes_for_graph.unsqueeze(0)

            # LSTM pooling
            out, (hn, cn) = self.lstm_pooling(nodes_for_graph)

            # Use the last LSTM output for this graph as its pooled representation
            pooled_representation = out[0, -1, :]
            pooled_outputs.append(pooled_representation)

        # Concatenate pooled representations for all graphs to match batch size
        x_pooled = torch.stack(pooled_outputs)

        x = self.mlp(x_pooled)
        return x