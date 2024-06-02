import torch
from dataset import load_and_preprocess_data, create_hetero_data
from model import HAN, train_xgboost_model
from utils import run_experiments, train_han_model, extract_embeddings
from option import parse_args

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metadata = (['circRNA', 'host_gene'], [('circRNA', 'interacts', 'host_gene'), ('host_gene', 'interacts', 'circRNA')])

    han_params = {
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'metadata': metadata,
        'heads1': args.heads1,
        'dropout1': args.dropout1,
        'heads2': args.heads2,
        'dropout2': args.dropout2,
    }

    train_params = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs
    }

    xgboost_params = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'early_stopping_rounds': args.early_stopping_rounds
    }

    run_experiments(
        n_experiments=args.n_experiments,
        load_and_preprocess_data=lambda: load_and_preprocess_data(args.train_path, args.val_path, args.test_path),
        create_hetero_data=create_hetero_data,
        HAN=HAN,
        train_han_model=train_han_model,
        train_xgboost_model=train_xgboost_model,
        device=device,
        metadata=metadata,
        han_params=han_params,
        xgboost_params=xgboost_params,
        train_params=train_params
    )

if __name__ == "__main__":
    main()
