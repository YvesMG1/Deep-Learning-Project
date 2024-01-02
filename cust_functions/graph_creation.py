
import torch
import networkx as nx

def create_pathway_graph(pathways, translation, descendants=True, delete_empty=True):
    
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
            G.nodes[pathway].setdefault('proteins', []).append(protein)
            if descendants:
                for descendant in descendant_map[pathway]:
                    G.nodes[descendant].setdefault('proteins', []).append(protein)
        
    if delete_empty:
        for node in list(G.nodes()):
            if 'proteins' not in G.nodes[node]:
                # Remove the node and its descendants if they don't have their own proteins
                nodes_to_remove = [node] + [n for n in descendant_map[node] if 'proteins' not in G.nodes[n]]
                G.remove_nodes_from(nodes_to_remove)
    return G


def map_edges_to_indices(edge_list):
    """ Maps a list of edges to a list of indices """

    # Create a unique mapping for each node
    unique_nodes = set([node for edge in edge_list for node in edge])
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    # Map edges to indices
    edge_index = [[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edge_list]
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
