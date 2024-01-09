import torch 
from torch_geometric.explain import *
from torch_geometric.explain import GNNExplainer
import matplotlib.pyplot as plt
import graph_creation 


def explain_function(model_train: torch.nn.Module, data: list):

    explainer = Explainer(
        model = model_train,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw',
        ),    
    #threshold_config=dict(threshold_type='topk', value=10), 
    )

    avg_feature_mask = torch.zeros((data[0].x.shape))
    avg_edge_mask = torch.zeros(data[0].edge_index.shape[1])
    
    for index, data_point in enumerate(data):
        explanation = explainer(x = data_point.x, edge_index = data_point.edge_index)
        avg_feature_mask = (avg_feature_mask * index + explanation.node_mask) / (index+1)
        avg_edge_mask = (avg_edge_mask * index + explanation.edge_mask) / (index+1)

    explanation_output = explainer(x = data[0].x, edge_index=data[0].edge_index)
    explanation_output.node_mask = avg_feature_mask
    explanation_output.edge_mask = avg_edge_mask

    return explanation_output


