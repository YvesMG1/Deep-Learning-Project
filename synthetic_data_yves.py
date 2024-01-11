import numpy as np
import pandas as pd

def create_synthetic_data():
    """
    return qm_matrix_synthetic,design_matrix_synthetic,translations_synthetic,parent_child_df
    """
    # Parameters
    num_patients = 500
    num_genes = 300
    num_key_genes = 20  # Number of genes particularly relevant to the disease
    num_pathways_per_level = [150, 50, 20, 10]
    num_key_pathways = 20  # Number of pathways particularly relevant to the disease

    # Simulate gene expression data
    gene_expression = np.random.rand(num_patients, num_genes)
    #gene_expression_noisy = gene_expression

    # Introduce noise
    gene_expression_noise = np.random.normal(0, 2, gene_expression.shape)
    gene_expression_noisy = gene_expression + gene_expression_noise
    
    
    # Introduce correlated noise for a subset of genes
    correlated_genes_indices = np.random.choice(num_genes, 200, replace=False)
    correlated_noise = np.random.normal(0, 0.5, (num_patients, 1))
    gene_expression_noisy[:, correlated_genes_indices] += correlated_noise
    #gene_expression_noisy = np.where(gene_expression_noisy < 0, 0, gene_expression_noisy)

    # Select key genes
    key_genes = np.random.choice(num_genes, num_key_genes, replace=False)
    
    """for key_gene in key_genes:
        gene_expression_noisy[:, key_gene] *= 2"""

    pathways = {}
    key_pathways = set()
    parent_child_pairs = []

    for level in range(4):
        for i in range(num_pathways_per_level[level]):
            pathway_name = f'Level_{level}Pathway{i}'
            if level == 0:
                pathway_genes = np.random.choice(num_genes, size=np.random.randint(5, 10), replace=False)
                pathways[pathway_name] = pathway_genes
            else:
                # Choose a random number of parents (for example, between 1 and 3)
                num_parents = np.random.randint(1, 3)
                for _ in range(num_parents):
                    parent_pathway = f'Level_{level-1}Pathway{np.random.randint(num_pathways_per_level[level-1])}'
                    parent_genes = pathways[parent_pathway]
                    num_genes_to_choose = min(len(parent_genes), np.random.randint(3, 10))
                    pathway_genes = np.random.choice(parent_genes, size=num_genes_to_choose, replace=False)

                    # Record each parent-child pair
                    parent_child_pairs.append((parent_pathway, pathway_name))

            pathways[pathway_name] = pathway_genes
            if i < num_key_pathways:
                key_pathways.add(pathway_name)

    # Creating DataFrame from parent-child pairs
    parent_child_df = pd.DataFrame(parent_child_pairs, columns=['parent', 'child'])



    # Now, parent_child_df contains the required data
    # Add noise pathways
    num_noise_pathways = 100
    for i in range(num_noise_pathways):
        pathways[f'Noise_Pathway_{i}'] = np.random.choice(num_genes, size=np.random.randint(5, 15), replace=False)

    def nonlinear_transform(expression):
        # Combining multiple mathematical functions for complexity
        #return np.sin(expression * np.pi) * np.tanh(expression) + np.cos(expression * np.pi / 2) * np.exp(expression)
        return np.log1p(np.abs(expression)) * np.exp(-expression**2) + np.sqrt(np.abs(expression + 0.5))
    # Function to simulate disease status with additional complexity
    def simulate_disease_status(gene_expression, pathways, key_pathways, key_genes):
        disease_status = []
        for patient in gene_expression:
            status = 0
            for pathway, genes in pathways.items():
                if 'Noise' not in pathway:
                    mean_expression = nonlinear_transform(np.mean(patient[genes]))
                    #print(mean_expression)
                    pathway_influence = 2 if pathway in key_pathways else 1
                    status += pathway_influence if mean_expression > 1 else 0
            print(np.mean(patient[key_genes]))
            key_gene_expression = nonlinear_transform(np.mean(patient[key_genes]))
            print(key_gene_expression)
            key_gene_influence = 20 if abs(key_gene_expression) > 1.2 else -1
            # Adjust overall threshold for disease status
            disease_threshold = len(pathways) / 3  # Adjusted to a fixed value
            # Influence of key genes
            status += key_gene_influence
            #print(status)
            print(status)
            disease_status.append(1 if status > disease_threshold else 0)
            
        return disease_status

    # Generate disease status
    disease_status = simulate_disease_status(gene_expression_noisy, pathways, key_pathways, key_genes)


    ######################## All good below ########################
    disease_status = [status + 1 for status in disease_status]
    # Create DataFrame
    df = pd.DataFrame(gene_expression_noisy, columns=[f'Gene_{i}' for i in range(num_genes)])
    df['group'] = disease_status

    qm_matrix_synthetic = df.iloc[:, :-1]
    design_matrix_synthetic = df.iloc[:, [-1]]
    translations_synthetic = pd.DataFrame([(number, pathway) for pathway, numbers in pathways.items() for number in numbers], columns=['input', 'translation'])
    
    # transform qm matrix
    qm_matrix_synthetic = qm_matrix_synthetic.T
    qm_matrix_synthetic.reset_index(inplace=True)
    qm_matrix_synthetic.rename(columns={'index': 'Genes'}, inplace=True)
    qm_matrix_synthetic.columns = ['Protein'] + [f'Patient{i-1}' for i in range(1, len(qm_matrix_synthetic.columns))]
    # transform design_matrix
    design_matrix_synthetic = design_matrix_synthetic.reset_index()
    design_matrix_synthetic.rename(columns={'index': 'sample'}, inplace=True)
    design_matrix_synthetic['sample'] = design_matrix_synthetic['sample'].apply(lambda x: f'Patient{x}')
    # Transorm translation
    #translations_synthetic.rename(columns={'Number': 'Gene'}, inplace=True)
    translations_synthetic["input"] = translations_synthetic["input"].apply(lambda x: f'Gene_{x}')
    return qm_matrix_synthetic,design_matrix_synthetic,translations_synthetic,parent_child_df,key_genes,key_pathways