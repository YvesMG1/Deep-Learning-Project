
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
import torch.nn.functional as F


class GATBlock(torch.nn.Module):
    """ GAT block with configurable number of heads, dropout rate, batch norm and residual connection """
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
    """ GCN block with configurable dropout rate, batch norm and residual connection """
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
    """ SAGE block with configurable dropout rate, batch norm and residual connection """
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
    """ Residual GCN with configurable number of layers, MLP and dropout rates """
    def __init__(self, num_features, layer_configs, mlp_config, num_classes):
        super(ResGCN, self).__init__()

        # GCN layers
        initial_layer = layer_configs[0]
        self.initial = GCNBlock(num_features, initial_layer['out_channels'], initial_layer['dropout_rate'], initial_layer['batch_norm'])

        self.hidden_layers = torch.nn.ModuleList()
        for layer_config in layer_configs[1:]:
            self.hidden_layers.append(GCNBlock(layer_config['in_channels'], layer_config['out_channels'], layer_config['dropout_rate'], layer_config['batch_norm'], residual=True))

        # MLP classifier
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

        # Loop over hidden layers and apply them sequentially
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        
        # Pooling
        x = global_max_pool(x, batch)

        # MLP classifier
        x = self.mlp(x)

        return x
    
class ResGAT(torch.nn.Module):
    """ Residual GAT with configurable number of layers, MLP and dropout rates """
    def __init__(self, num_features, layer_configs, mlp_config, num_classes):
        super(ResGAT, self).__init__()

        # GAT layers
        initial_layer = layer_configs[0]
        self.initial = GATBlock(num_features, initial_layer['out_channels'], initial_layer.get('heads', 1), initial_layer['dropout_rate'], initial_layer['batch_norm'])

        self.hidden_layers = torch.nn.ModuleList()
        for layer_config in layer_configs[1:]:
            self.hidden_layers.append(GATBlock(layer_config['in_channels'], layer_config['out_channels'], layer_config.get('heads', 1), layer_config['dropout_rate'], layer_config['batch_norm'], residual=True))

        # MLP classifier
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

        # Loop over hidden layers and apply them sequentially
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        # Pooling 
        x = global_max_pool(x, batch)

        # MLP classifier
        x = self.mlp(x)

        return x
