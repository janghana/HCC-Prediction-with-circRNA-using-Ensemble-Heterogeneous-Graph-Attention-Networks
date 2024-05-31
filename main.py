import argparse
import torch
from dataset import CircRNADataset
from models import HANModel
from train import train_model, evaluate_metrics, get_embeddings_and_attention
import xgboost as xgb
import numpy as np
from visualization import plot_tsne, plot_heatmap

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 로드
    dataset = CircRNADataset(args.train_path, args.val_path, args.test_path)
    train_data, val_data, test_data = dataset.get_data()

    def get_embeddings(data):
        return data['circRNA'].x.numpy(), data['circRNA'].y.numpy()

    train_embeddings, train_labels = get_embeddings(train_data)
    val_embeddings, val_labels = get_embeddings(val_data)
    test_embeddings, test_labels = get_embeddings(test_data)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model_path)
    
    train_preds = xgb_model.predict_proba(train_embeddings)[:, 1]
    val_preds = xgb_model.predict_proba(val_embeddings)[:, 1]
    test_preds = xgb_model.predict_proba(test_embeddings)[:, 1]

    xgb_test_preds = xgb_model.predict(test_embeddings)
    xgb_acc, xgb_precision, xgb_recall, xgb_f1 = evaluate_metrics(test_labels, xgb_test_preds)
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    print(f"XGBoost Test Precision: {xgb_precision:.4f}")
    print(f"XGBoost Test Recall: {xgb_recall:.4f}")
    print(f"XGBoost Test F1 Score: {xgb_f1:.4f}")

    model = HANModel(hidden_channels=args.hidden_channels, out_channels=args.out_channels, num_heads=args.num_heads, num_layers=args.num_layers).to(device)
    model = train_model(model, train_data, val_data, train_preds, device)

    model.eval()
    test_data = test_data.to(device)
    x_dict = {
        'circRNA': test_data['circRNA'].x,
        'gene': test_data['gene'].x
    }
    
    _, attention_weights = model(x_dict, test_data.edge_index_dict, return_attention_weights=True)

    positive_indices = test_labels == 1
    negative_indices = test_labels == 0

    positive_attention_weights = attention_weights[0][positive_indices]
    negative_attention_weights = attention_weights[0][negative_indices]

    top_20_positive_circRNAs = np.argsort(positive_attention_weights.sum(axis=1))[-20:]
    top_20_positive_host_genes = np.argsort(positive_attention_weights.sum(axis=0))[-20:]

    top_20_negative_circRNAs = np.argsort(negative_attention_weights.sum(axis=1))[-20:]
    top_20_negative_host_genes = np.argsort(negative_attention_weights.sum(axis=0))[-20:]

    top_20_positive_circRNA_names = [dataset.circRNA_names[i] for i in top_20_positive_circRNAs]
    top_20_positive_host_gene_names = [dataset.gene_names[i] for i in top_20_positive_host_genes]

    top_20_negative_circRNA_names = [dataset.circRNA_names[i] for i in top_20_negative_circRNAs]
    top_20_negative_host_gene_names = [dataset.gene_names[i] for i in top_20_negative_host_genes]

    filtered_positive_attention = positive_attention_weights[top_20_positive_circRNAs][:, top_20_positive_host_genes]
    filtered_negative_attention = negative_attention_weights[top_20_negative_circRNAs][:, top_20_negative_host_genes]

    total_attention_score = attention_weights[0].sum(axis=1).sum()

    top_6_attention_indices = np.argsort(attention_weights[0].sum(axis=1))[-6:]
    top_6_attention_circRNA_names = [dataset.circRNA_names[i] for i in top_6_attention_indices]
    top_6_attention_scores = attention_weights[0].sum(axis=1)[top_6_attention_indices]

    print("Top 6 circRNAs with highest attention scores (as percentages):")
    for name, score in zip(top_6_attention_circRNA_names, top_6_attention_scores):
        percentage = (score / total_attention_score) * 100
        print(f"{name}: {percentage:.2f}%")

    bottom_6_attention_indices = np.argsort(attention_weights[0].sum(axis=1))[:6]
    bottom_6_attention_circRNA_names = [dataset.circRNA_names[i] for i in bottom_6_attention_indices]
    bottom_6_attention_scores = attention_weights[0].sum(axis=1)[bottom_6_attention_indices]

    print("Bottom 6 circRNAs with lowest attention scores (as percentages):")
    for name, score in zip(bottom_6_attention_circRNA_names, bottom_6_attention_scores):
        percentage = (score / total_attention_score) * 100
        print(f"{name}: {percentage:.2f}%")

    plot_heatmap(filtered_positive_attention, top_20_positive_circRNA_names, top_20_positive_host_gene_names, 'Positive circRNAs and Host Genes Heatmap', 'HAN_XGBOOST_heatmap_positive.png', 'Reds')
    plot_heatmap(filtered_negative_attention, top_20_negative_circRNA_names, top_20_negative_host_gene_names, 'Negative circRNAs and Host Genes Heatmap', 'HAN_XGBOOST_heatmap_negative.png', 'Blues')
    plot_tsne(test_embeddings, test_labels, 'HAN_XGBOOST_tsne.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate HAN model with XGBoost knowledge distillation")
    parser.add_argument('--train_path', type=str, default='./dataset/re_train.csv', help='Path to the training data')
    parser.add_argument('--val_path', type=str, default='./dataset/re_val.csv', help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default='./dataset/re_test.csv', help='Path to the test data')
    parser.add_argument('--xgb_model_path', type=str, default='xgboost_model_5.json', help='Path to the pre-trained XGBoost model')
    parser.add_argument('--hidden_channels', type=int, default=512, help='Number of hidden channels')
    parser.add_argument('--out_channels', type=int, default=2, help='Number of output channels')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    
    args = parser.parse_args()
    main(args)
