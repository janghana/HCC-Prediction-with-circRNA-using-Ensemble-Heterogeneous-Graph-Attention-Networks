import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from statistics import mean, stdev
import os
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import shutil

def train_han_model(model, train_loader, val_loader, device, optimizer, epochs, save_path=None):
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            loss = F.cross_entropy(out['circRNA'], data['circRNA'].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict)
            pred = out['circRNA'].max(1)[1]
            correct += pred.eq(data['circRNA'].y).sum().item()
            total += data['circRNA'].y.size(0)
        return correct / total

    for epoch in range(1, epochs + 1):
        train()
        val_acc = evaluate(val_loader)
        print(f'Epoch: {epoch:03d}, Val Acc: {val_acc:.4f}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    return model

def extract_embeddings(model, data, device):
    model.eval()
    with torch.no_grad():
        data.to(device)
        embeddings = model(data.x_dict, data.edge_index_dict)['circRNA'].cpu().numpy()
    return embeddings

def run_experiments_cv(load_and_preprocess_data, create_hetero_data, HAN, train_han_model, train_xgboost_model, device, metadata, han_params, xgboost_params, train_params):
    accuracies = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    best_accuracy = 0.0
    best_fold = 0
    best_han_model_path = None
    best_xgboost_model_path = None

    train_features, train_labels, val_features, val_labels, test_features, test_labels, train_circRNA_nodes, train_host_gene_nodes, val_circRNA_nodes, val_host_gene_nodes, test_circRNA_nodes, test_host_gene_nodes = load_and_preprocess_data()

    all_features = np.vstack([train_features, val_features])
    all_labels = np.hstack([train_labels, val_labels])
    all_circRNA_nodes = np.hstack([train_circRNA_nodes, val_circRNA_nodes])
    all_host_gene_nodes = np.hstack([train_host_gene_nodes, val_host_gene_nodes])

    # KFold for 5-fold cross-validation
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    fold_idx = 1
    for train_index, val_index in kf.split(all_features):
        print(f"Running fold {fold_idx}...")

        train_features_fold, val_features_fold = all_features[train_index], all_features[val_index]
        train_labels_fold, val_labels_fold = all_labels[train_index], all_labels[val_index]
        train_circRNA_nodes_fold, val_circRNA_nodes_fold = all_circRNA_nodes[train_index], all_circRNA_nodes[val_index]
        train_host_gene_nodes_fold, val_host_gene_nodes_fold = all_host_gene_nodes[train_index], all_host_gene_nodes[val_index]

        train_data = create_hetero_data(train_features_fold, train_circRNA_nodes_fold, train_host_gene_nodes_fold, train_labels_fold)
        val_data = create_hetero_data(val_features_fold, val_circRNA_nodes_fold, val_host_gene_nodes_fold, val_labels_fold)

        train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
        val_loader = DataLoader([val_data], batch_size=1)

        model = HAN(**han_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

        model_save_path = f'./saved_models/han_model_fold_{fold_idx}.pth'

        model = train_han_model(model, train_loader, val_loader, device, optimizer, train_params['epochs'], save_path=model_save_path)

        train_embeddings = extract_embeddings(model, train_data, device)
        val_embeddings = extract_embeddings(model, val_data, device)

        accuracy, report, xgb_model = train_xgboost_model(train_embeddings, train_labels_fold, val_embeddings, val_labels_fold, **xgboost_params)
        
        xgboost_model_save_path = f'./saved_models/xgboost_model_fold_{fold_idx}.pkl'
        joblib.dump(xgb_model, xgboost_model_save_path)  # Save the model
        print(f'XGBoost model saved to {xgboost_model_save_path}')

        accuracies.append(accuracy)
        precision_scores.append(report['weighted avg']['precision'])
        recall_scores.append(report['weighted avg']['recall'])
        f1_scores.append(report['weighted avg']['f1-score'])

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_fold = fold_idx
            best_han_model_path = model_save_path
            best_xgboost_model_path = xgboost_model_save_path

        fold_idx += 1

    print('5-Fold Cross-Validation Results:')
    print(f'Accuracy: {mean(accuracies):.4f} ± {stdev(accuracies):.4f}')
    print(f'Precision: {mean(precision_scores):.4f} ± {stdev(precision_scores):.4f}')
    print(f'Recall: {mean(recall_scores):.4f} ± {stdev(recall_scores):.4f}')
    print(f'F1-Score: {mean(f1_scores):.4f} ± {stdev(f1_scores):.4f}')

    if best_han_model_path:
        shutil.copy(best_han_model_path, './saved_models/best_han_model.pth')
        print(f'Best HAN model saved as: ./saved_models/best_han_model.pth')

    if best_xgboost_model_path:
        shutil.copy(best_xgboost_model_path, './saved_models/best_xgboost_model.pkl')
        print(f'Best XGBoost model saved as: ./saved_models/best_xgboost_model.pkl')

    print(f'Best fold: {best_fold} with accuracy: {best_accuracy:.4f}')
    print(f'Best HAN model saved at: ./saved_models/best_han_model.pth')
    print(f'Best XGBoost model saved at: ./saved_models/best_xgboost_model.pkl')

    return './saved_models/best_han_model.pth', './saved_models/best_xgboost_model.pkl'

def load_xgboost_model(model_path):
    """Load the pre-trained XGBoost model from a file."""
    xgb_model = joblib.load(model_path)
    return xgb_model

def run_xgboost_inference(xgb_model, test_embeddings, test_labels):
    """Perform inference using a pre-trained XGBoost model."""
    test_preds = xgb_model.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds)
    return accuracy, report, test_preds

def load_model(model, model_path, device):
    """Load the saved HAN model from a file."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
