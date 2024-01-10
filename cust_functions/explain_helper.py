import torch 
from torch_geometric.explain import *
from torch_geometric.explain import GNNExplainer
import matplotlib.pyplot as plt

import pandas as pd 
from .graph_creation import create_pathway_graph

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


def importance_calculator(explanation: Explanation, node: bool, k_node: int, feature: bool, k_feature: int, input_data_preprocessed, pathways, translation):

    '''
    The explanation model has an iomportance matrix (num_nodes, num_features)
    The feature importance is then calculated as the sum of a feature across all nodes
    The node importance is calculated as the sum of a node across all features
    '''


    top_nodes, top_features = pd.DataFrame, pd.DataFrame
    
    # Turn node_mask into pandas 
    node_mask = explanation.node_mask 
    pd_node_mask = pd.DataFrame(node_mask.numpy())
    
    # Retrieve k_node most important nodes
    if (node == True):
        pd_node_mask['Node_score'] = pd_node_mask.sum(axis=1) # Per node sum horizontally across all features
        top_nodes = pd_node_mask.nlargest(k_node, 'Node_score')
        top_nodes = top_nodes.drop(columns=top_nodes.columns.difference(['Node_score']))
        pd_node_mask = pd_node_mask.drop('Node_score', axis=1)

    if (feature == True): 
        pd_node_mask['Feature_score'] = pd_node_mask.sum(axis=0) # Per feature sum vertically across all nodes
        top_features = pd_node_mask.nlargest(k_feature, 'Feature_score')
        top_features = top_features.drop(columns=top_features.columns.difference(['Feature_score']))

    # Connect node indices to their pathway name
    sample_graph = create_pathway_graph(pathways, translation, descendants=True)
    index_node_match = {'Pathway': list(sample_graph.nodes())}
    index_node_match = pd.DataFrame(index_node_match)
    top_nodes = pd.merge(top_nodes, index_node_match, left_index=True, right_index=True, how='inner')
        
    # Connect feature indices to their protein name
    index_protein_match = input_data_preprocessed.drop(columns=input_data_preprocessed.columns.difference(['Protein']))
    top_features = pd.merge(top_features, index_protein_match, left_index=True, right_index=True, how='inner')


    return top_nodes, top_features



def explain_wrapper(model_explain_init: torch.nn.Module, k_top: int, path: str, explain_data: list, structural_data: list, device):

    '''
    Returns: Two pandas Dataframes, one for the most important features and one for the most important nodes
    '''

    relative_path = "./trained_models/" + path

    model_explain_init.load_state_dict(torch.load(relative_path, map_location=torch.device(device)))
    model_explain_init.eval()

    explanation = explain_function(model_explain_init, explain_data)

    top_nodes, top_features = importance_calculator(explanation, 
                                                    True, k_top, True, k_top, 
                                                    structural_data[0], structural_data[1], structural_data[2])
    

    return top_nodes, top_features
