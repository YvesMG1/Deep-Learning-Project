
import torch
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import random
from torch_geometric.data import Data

def create_pathway_graph(pathways, translation, descendants=True, perturb = False, edge_removal_prob=0.5, edge_addition_prob=0.5):
    
    G = nx.DiGraph()
    for _, row in pathways.iterrows():
        G.add_edge(row['parent'], row['child'])

    descendant_map = {}
    if descendants:
        for node in G.nodes():
            descendant_map[node] = set(nx.descendants(G, node))

    for _, row in translation.iterrows():
        protein = row['input']
        pathway = row['translation']
        if pathway in G:
            # create node for protein if it doesn't exist
            if 'proteins' not in G.nodes[pathway]:
                G.nodes[pathway]['proteins'] = []
            # check if protein is in pathway
            if protein not in G.nodes[pathway]['proteins']:
                G.nodes[pathway]['proteins'].append(protein)
            
            # check if pathway has descendants
            if len(descendant_map[pathway]) != 0:
                for descendant in descendant_map[pathway]:
                    # create node for protein if it doesn't exist
                    if 'proteins' not in G.nodes[descendant]:
                        G.nodes[descendant]['proteins'] = []
                    # check if protein is in pathway
                    if protein not in G.nodes[descendant]['proteins']:
                        G.nodes[descendant]['proteins'].append(protein)

    if perturb:
        G = perturb_graph(G, edge_removal_prob=edge_removal_prob, edge_addition_prob=edge_addition_prob)

    return G


def map_edges_to_indices(edge_list):
    """ Maps a list of edges to a list of indices """

    # Create a unique mapping for each node
    unique_nodes = set([node for edge in edge_list for node in edge])
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    # Map edges to indices
    edge_index = [[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edge_list]
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def perturb_graph(graph, edge_removal_prob=0.5, edge_addition_prob=0.5):
    # Copy the original graph
    perturbed_graph = graph.copy()

    # Random Edge Removal
    edges = list(perturbed_graph.edges())
    num_edges_to_remove = int(edge_removal_prob * len(edges))
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    perturbed_graph.remove_edges_from(edges_to_remove)

    # Random Edge Addition
    nodes = list(perturbed_graph.nodes())
    possible_new_edges = [(i, j) for i in nodes for j in nodes if i != j and not perturbed_graph.has_edge(i, j)]
    num_edges_to_add = int(edge_addition_prob * len(possible_new_edges))
    edges_to_add = random.sample(possible_new_edges, num_edges_to_add)
    perturbed_graph.add_edges_from(edges_to_add)

    return perturbed_graph

def create_patient_feature_dic(design_matrix, input_data_preprocessed, graph, gen_column = 'Protein'):

    patient_ids = design_matrix['sample'].values
    patient_features = {patient_id: [] for patient_id in patient_ids}
    protein_names = input_data_preprocessed[gen_column].tolist()

    for patient_id in patient_ids:
        pathway_features = np.zeros((len(graph.nodes()), len(protein_names)))
        for i, pathway in enumerate(graph.nodes()):
            proteins_in_pathway = graph.nodes[pathway].get('proteins', [])
            for j, protein in enumerate(protein_names):
                if protein in proteins_in_pathway:
                    pathway_features[i, j] = input_data_preprocessed.loc[input_data_preprocessed[gen_column] == protein, patient_id].values[0]

        patient_features[patient_id] = pathway_features

    return patient_features

def pytorch_graphdata(design_matrix, data, graph, gen_column = 'Protein', load_data = True, save_data = False, path = 'pytorch_data/graph_data.pkl'):

    graph_data = None
    graph_data_list = None

    if load_data:
        with open(path, 'rb') as f:
            graph_data_list = pickle.load(f)
    else:
        patient_features = create_patient_feature_dic(design_matrix, data, graph, gen_column = gen_column)
        patient_ids = design_matrix['sample'].values

        graph_data_list = []

        for patient_id in patient_ids:
            patient_graph = graph.copy()

            # Get the PCA-reduced features for this patient
            features = patient_features[patient_id]

            # Convert NetworkX graph to PyTorch Geometric Data
            edge_index = map_edges_to_indices(patient_graph.edges)
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor([design_matrix[design_matrix['sample'] == patient_id]['group'].iloc[0]], dtype=torch.long)
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            graph_data_list.append(graph_data)
            
        if save_data:
            with open(path, 'wb') as f:
                pickle.dump(graph_data_list, f)
                
    return graph_data_list
