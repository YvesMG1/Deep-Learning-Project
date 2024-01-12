import torch 
from torch_geometric.explain import *
from torch_geometric.explain import GNNExplainer
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd 
import pickle 


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


def importance_calculator(explanation: Explanation, input_data_preprocessed, graph):

    '''
    The explanation model has an iomportance matrix (num_nodes, num_features)
    The feature importance is then calculated as the sum of a feature across all nodes
    The node importance is calculated as the sum of a node across all features
    '''


    top_nodes, top_features = pd.DataFrame, pd.DataFrame
    
    # Turn node_mask into pandas 
    node_mask = explanation.node_mask 
    pd_node_mask = pd.DataFrame(node_mask.numpy())
    
    # Retrieve most important nodes
    pd_node_mask['Node_score'] = pd_node_mask.sum(axis=1) # Per node sum horizontally across all features
    top_nodes = pd_node_mask.sort_values(by = 'Node_score', ascending=False)
    top_nodes = top_nodes.drop(columns=top_nodes.columns.difference(['Node_score']))
    pd_node_mask = pd_node_mask.drop('Node_score', axis=1)

    # Retrieve most important features
    pd_node_mask['Feature_score'] = pd_node_mask.sum(axis=0) # Per feature sum vertically across all nodes
    top_features = pd_node_mask.sort_values(by = 'Feature_score', ascending=False)
    top_features = top_features.drop(columns=top_features.columns.difference(['Feature_score']))

    # Connect node indices to their pathway name
    index_node_match = {'Pathway': list(graph.nodes())}
    index_node_match = pd.DataFrame(index_node_match)
    top_nodes = pd.merge(top_nodes, index_node_match, left_index=True, right_index=True, how='inner')
        
    # Connect feature indices to their protein name
    index_protein_match = input_data_preprocessed.drop(columns=input_data_preprocessed.columns.difference(['Protein']))
    top_features = pd.merge(top_features, index_protein_match, left_index=True, right_index=True, how='inner')


    return top_nodes, top_features



def explain_wrapper(model_explain_init: torch.nn.Module, path: str, explain_data: list, structural_data: list, device):

    '''
    Returns: Two pandas Dataframes, one for the most important features and one for the most important nodes
    '''

    relative_path = "./trained_models/" + path

    model_explain_init.load_state_dict(torch.load(relative_path, map_location=torch.device(device)))
    model_explain_init.eval()

    explanation = explain_function(model_explain_init, explain_data)

    top_nodes, top_features = importance_calculator(explanation, 
                                                    structural_data[0], structural_data[1])
    

    return top_nodes, top_features


def create_avg_sd_df (path: str, retrieve_feature: bool, sort: bool): 

    if retrieve_feature == True: # Receive Feature
        position_dict_list = 1
        access = 'Feature_score'
        avg_score = 'Avg_Feature_Score'
        sd_score = 'Sd_Feature_score'
    else: # Receive nodes
        position_dict_list = 0
        access = 'Node_score'
        avg_score = 'Avg_Node_Score'
        sd_score = 'Sd_Node_score'

    with open(path, 'rb') as file:
        # Load the dataset using pickle.load
        Top_feature_nodes_per_fold = pickle.load(file)

    features_or_nodes_list = []

    for key, value in Top_feature_nodes_per_fold.items():
        features_or_nodes_list.append(value[position_dict_list].sort_index()) # The 1 retrieves the feature scores instead of the node scores

    avg_FeaturesOrNodes_across_folds = features_or_nodes_list[0] # The Dataframe we will store average and standard deviation across all folds in 
    avg_FeaturesOrNodes_across_folds[access] = 0
    avg_FeaturesOrNodes_across_folds = avg_FeaturesOrNodes_across_folds.rename(columns={access: avg_score})
    avg_FeaturesOrNodes_across_folds = avg_FeaturesOrNodes_across_folds[avg_FeaturesOrNodes_across_folds.columns[::-1]]
    avg_FeaturesOrNodes_across_folds[sd_score] = 0

    for fold in features_or_nodes_list: # Each element is a pandas Df with protein/feature importance ordered by index 
        if fold.isna().any().any():
            print("DataFrame contains NaN values")
        avg_FeaturesOrNodes_across_folds[avg_score] = avg_FeaturesOrNodes_across_folds[avg_score] + fold[access]
        avg_FeaturesOrNodes_across_folds[sd_score] = avg_FeaturesOrNodes_across_folds[sd_score] + np.square(fold[access])

    avg_FeaturesOrNodes_across_folds[avg_score] = avg_FeaturesOrNodes_across_folds[avg_score] / len(features_or_nodes_list)
    avg_FeaturesOrNodes_across_folds[sd_score] = avg_FeaturesOrNodes_across_folds[sd_score] / len(features_or_nodes_list)

    avg_FeaturesOrNodes_across_folds[sd_score] = np.sqrt(avg_FeaturesOrNodes_across_folds[sd_score] - np.square(avg_FeaturesOrNodes_across_folds[avg_score]))

    if sort:
        return avg_FeaturesOrNodes_across_folds.sort_values(by=avg_score, ascending=False)
    else: 
        return avg_FeaturesOrNodes_across_folds
