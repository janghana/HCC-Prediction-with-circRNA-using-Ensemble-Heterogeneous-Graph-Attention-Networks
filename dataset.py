import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData

class CircRNADataset:
    def __init__(self, train_path, val_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.val_data = pd.read_csv(val_path)
        self.test_data = pd.read_csv(test_path)
        self.label_encoder = LabelEncoder()

        self.train_data['label'] = self.label_encoder.fit_transform(self.train_data['label'])
        self.val_data['label'] = self.label_encoder.fit_transform(self.val_data['label'])
        self.test_data['label'] = self.label_encoder.fit_transform(self.test_data['label'])

        self.circRNA_names = list(self.train_data['circRNA'].unique())
        self.gene_names = list(self.train_data['host_gene'].unique())

    def process(self, data):
        hetero_data = HeteroData()

        circRNA_ids = data['circRNA'].unique()
        gene_ids = data['host_gene'].unique()

        circRNA_index = {id: idx for idx, id in enumerate(circRNA_ids)}
        gene_index = {id: idx for idx, id in enumerate(gene_ids)}

        hetero_data['circRNA'].num_nodes = len(circRNA_ids)
        hetero_data['gene'].num_nodes = len(gene_ids)

        circRNA_to_gene = [(circRNA_index[row['circRNA']], gene_index[row['host_gene']])
                           for idx, row in data.iterrows()]
        gene_to_circRNA = [(gene_index[row['host_gene']], circRNA_index[row['circRNA']])
                           for idx, row in data.iterrows()]

        hetero_data['circRNA', 'interacts_with', 'gene'].edge_index = torch.tensor(circRNA_to_gene, dtype=torch.long).t().contiguous()
        hetero_data['gene', 'rev_interacts_with', 'circRNA'].edge_index = torch.tensor(gene_to_circRNA, dtype=torch.long).t().contiguous()

        hetero_data['circRNA'].x = torch.tensor(data[['average_log2SRPTM', 'average_read_counts']].values, dtype=torch.float)
        hetero_data['gene'].x = torch.tensor(data[['position_Exon', 'position_Intron', 'gene_type_lncRNA', 'gene_type_mRNA']].values, dtype=torch.float)

        hetero_data['circRNA'].y = torch.tensor(data['label'].values, dtype=torch.long)

        return hetero_data

    def get_data(self):
        train_data = self.process(self.train_data)
        val_data = self.process(self.val_data)
        test_data = self.process(self.test_data)

        return train_data, val_data, test_data
