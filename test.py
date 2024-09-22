import torch
import joblib
from model import HAN
from option import parse_args
from utils import extract_embeddings, load_xgboost_model, run_xgboost_inference, load_model
from dataset import load_and_preprocess_data, create_hetero_data
import numpy as np


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metadata = (['circRNA', 'host_gene'], [('circRNA', 'interacts', 'host_gene'), ('host_gene', 'interacts', 'circRNA')])

    # Load and preprocess data
    _, _, _, _, test_features, test_labels, _, _, _, _, test_circRNA_nodes, test_host_gene_nodes = load_and_preprocess_data(
        args.train_path, args.val_path, args.test_path
    )
    
    # Create hetero data for test set
    test_data = create_hetero_data(test_features, test_circRNA_nodes, test_host_gene_nodes, test_labels)

    # Initialize and load HAN model
    model = HAN(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        metadata=metadata,
        heads1=args.heads1,
        dropout1=args.dropout1,
        heads2=args.heads2,
        dropout2=args.dropout2
    ).to(device)
    model = load_model(model, './saved_models/best_han_model.pth', device)

    test_embeddings = extract_embeddings(model, test_data, device)
    xgb_model = load_xgboost_model('./saved_models/best_xgboost_model.pkl')

    _ , report, _ = run_xgboost_inference(xgb_model, test_embeddings, test_labels)

    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
