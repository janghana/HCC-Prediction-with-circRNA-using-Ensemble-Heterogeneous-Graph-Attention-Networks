import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

class HAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads1, dropout1, heads2, dropout2):
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, out_channels, metadata=metadata, heads=heads1, dropout=dropout1, cat=False)
        self.conv2 = HANConv(out_channels, out_channels, metadata=metadata, heads=heads2, dropout=dropout2, cat=False)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

def train_xgboost_model(train_embeddings, train_labels, test_embeddings, test_labels, n_estimators, learning_rate, max_depth, subsample, colsample_bytree, early_stopping_rounds):
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds
    )
    xgb_model.fit(
        train_embeddings, train_labels,
        eval_set=[(train_embeddings, train_labels), (test_embeddings, test_labels)],
        verbose=True
    )
    test_preds = xgb_model.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, output_dict=True)
    return accuracy, report
