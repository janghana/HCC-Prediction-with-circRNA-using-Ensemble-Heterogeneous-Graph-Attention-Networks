import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from statistics import mean, stdev

def train_han_model(model, train_loader, val_loader, device, optimizer, epochs):
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
    
    return model

def extract_embeddings(model, data, device):
    model.eval()
    with torch.no_grad():
        data.to(device)
        embeddings = model(data.x_dict, data.edge_index_dict)['circRNA'].cpu().numpy()
    return embeddings

def run_experiments(n_experiments, load_and_preprocess_data, create_hetero_data, HAN, train_han_model, train_xgboost_model, device, metadata, han_params, xgboost_params, train_params):
    accuracies = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for _ in range(n_experiments):
        train_features, train_labels, val_features, val_labels, test_features, test_labels, train_circRNA_nodes, train_host_gene_nodes, val_circRNA_nodes, val_host_gene_nodes, test_circRNA_nodes, test_host_gene_nodes = load_and_preprocess_data()

        train_data = create_hetero_data(train_features, train_circRNA_nodes, train_host_gene_nodes, train_labels)
        val_data = create_hetero_data(val_features, val_circRNA_nodes, val_host_gene_nodes, val_labels)
        test_data = create_hetero_data(test_features, test_circRNA_nodes, test_host_gene_nodes, test_labels)

        train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
        val_loader = DataLoader([val_data], batch_size=1)
        test_loader = DataLoader([test_data], batch_size=1)

        model = HAN(**han_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

        model = train_han_model(model, train_loader, val_loader, device, optimizer, train_params['epochs'])

        train_embeddings = extract_embeddings(model, train_data, device)
        test_embeddings = extract_embeddings(model, test_data, device)

        accuracy, report = train_xgboost_model(train_embeddings, train_labels, test_embeddings, test_labels, **xgboost_params)
        accuracies.append(accuracy)
        precision_scores.append(report['weighted avg']['precision'])
        recall_scores.append(report['weighted avg']['recall'])
        f1_scores.append(report['weighted avg']['f1-score'])

    print(f'Accuracy: {mean(accuracies):.4f} ± {stdev(accuracies):.4f}')
    print(f'Precision: {mean(precision_scores):.4f} ± {stdev(precision_scores):.4f}')
    print(f'Recall: {mean(recall_scores):.4f} ± {stdev(recall_scores):.4f}')
    print(f'F1-Score: {mean(f1_scores):.4f} ± {stdev(f1_scores):.4f}')
