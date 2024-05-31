import torch
from torch.nn import MSELoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def train_model(model, train_data, val_data, train_preds, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.00001)

    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        train_data = train_data.to(device)
        print(epoch)
        x_dict = {
            'circRNA': train_data['circRNA'].x,
            'gene': train_data['gene'].x
        }
        
        out, attention_weights = model(x_dict, train_data.edge_index_dict, return_attention_weights=True)
        out = out['circRNA'][:, 1]  # Get probability of class 1
        loss = criterion(out, torch.tensor(train_preds, dtype=torch.float, device=device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    return model

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def get_embeddings_and_attention(model, data, device):
    model.eval()
    data = data.to(device)
    
    x_dict = {
        'circRNA': data['circRNA'].x,
        'gene': data['gene'].x
    }
    
    embeddings, attention_weights = model.get_node_embeddings(x_dict, data.edge_index_dict)
    return embeddings['circRNA'].cpu().detach().numpy(), data['circRNA'].y.cpu().detach().numpy(), attention_weights
