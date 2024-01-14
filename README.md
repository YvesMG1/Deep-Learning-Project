# Comparing GNNs and MLPs in the context of 'biologically informed' disease prediction and biomarker identification 
Deep Learning Project 2024 - by Frederick Geiger, Yves Görgen and Hendrik Plett


## 1. Data 

Under './aki_data' one finds the data for the Acute Kidney Injury (AKI) and the structural pathway information. This data was obtained from the GitHub repository belonging the paper Hartman et al. (2023). <br>

Under './covid_data' one finds the dataset for COVID-19. This dataset is not publicy available and was acquired by reaching out to Hartman. 

## 2. Code 

### 2.0 Graph creation

The file './cust_functions/graph_creation.py' provides the function needed to turn the raw data from 1. into graph-structured data that can be fed into the models.

### 2.1. GNNs

The file './cust_functions/graph_networks.py' defines the ResGCN and ResGAT architecture. <br> 
The file './GNN_implementation_v3_yves.ipynb' then creates the graph data and trains the GNN models on it. <br> 
For training and evaluation, the file './cust_functions/training.py' offers auxiliary functions.

### 2.2 GNN Interpretation 

The file './cust_functions/explain_helper.py' provides necessary functions to detect important features and nodes in our GNNs. <br>
The file './Explain.ipynb' then computes the key features and nodes. <br>
The file './Explain_statistics.ipynb' brings together the GNN, BINN and ML interpreations and provides summary statistics. 

### 2.3. BINN



### 2.4 ML-models

The file './ML_models' trains RandomForest, AdaBoost and Logistical Regression. Also it directly computes the important features. 

## 3. Dependencies 

The code only runs (without modification) if the following dependencies are installed: 
