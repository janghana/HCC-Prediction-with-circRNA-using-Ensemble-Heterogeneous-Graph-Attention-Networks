import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import HeteroData

def load_and_preprocess_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    def preprocess_data(df):
        features = df[['average_log2SRPTM', 'average_read_counts']].values
        labels = df['label'].values
        circRNA_nodes = df['circRNA'].values
        host_gene_nodes = df['host_gene'].values
        return features, labels, circRNA_nodes, host_gene_nodes

    train_features, train_labels, train_circRNA_nodes, train_host_gene_nodes = preprocess_data(train_df)
    val_features, val_labels, val_circRNA_nodes, val_host_gene_nodes = preprocess_data(val_df)
    test_features, test_labels, test_circRNA_nodes, test_host_gene_nodes = preprocess_data(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels, train_circRNA_nodes, train_host_gene_nodes, val_circRNA_nodes, val_host_gene_nodes, test_circRNA_nodes, test_host_gene_nodes

def load_and_preprocess_data_test(test_path):
    test_df = pd.read_csv(test_path)

    def preprocess_data(df):
        features = df[['average_log2SRPTM', 'average_read_counts']].values
        labels = df['label'].values
        circRNA_nodes = df['circRNA'].values
        host_gene_nodes = df['host_gene'].values
        return features, labels, circRNA_nodes, host_gene_nodes

    test_features, test_labels, test_circRNA_nodes, test_host_gene_nodes = preprocess_data(test_df)

    scaler = StandardScaler()
    test_features = scaler.fit_transform(test_features)

    return test_features, test_labels, test_circRNA_nodes, test_host_gene_nodes

def create_hetero_data(features, circRNA_nodes, host_gene_nodes, labels):
    data = HeteroData()
    data['circRNA'].x = torch.tensor(features, dtype=torch.float)
    data['host_gene'].x = torch.tensor(features, dtype=torch.float)
    data['circRNA', 'interacts', 'host_gene'].edge_index = torch.tensor(
        [range(len(circRNA_nodes)), range(len(host_gene_nodes))], dtype=torch.long)
    data['host_gene', 'interacts', 'circRNA'].edge_index = torch.tensor(
        [range(len(host_gene_nodes)), range(len(circRNA_nodes))], dtype=torch.long)
    data['circRNA'].y = torch.tensor(labels, dtype=torch.long)
    return data
