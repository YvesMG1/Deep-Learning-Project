
import torch
import networkx as nx

def create_pathway_graph(pathways, translation, descendants=True):
    
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

    return G


def map_edges_to_indices(edge_list):
    """ Maps a list of edges to a list of indices """

    # Create a unique mapping for each node
    unique_nodes = set([node for edge in edge_list for node in edge])
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    # Map edges to indices
    edge_index = [[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edge_list]
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
