from sklearn.model_selection import KFold
import torch
from dataset import load_and_preprocess_data, create_hetero_data
from model import HAN, train_xgboost_model
from utils import run_experiments_cv, train_han_model
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

    # Perform 5-fold cross-validation
    run_experiments_cv(
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

# python train.py --train_path ./dataset/train.csv --val_path ./dataset/val.csv --test_path ./dataset/inference.csv --n_experiments 5 --in_channels 2 --out_channels 64 --heads1 8 --dropout1 0.6 --heads2 1 --dropout2 0.6 --lr 0.05 --weight_decay 0.0005 --epochs 50 --n_estimators 50000 --learning_rate 0.5 --max_depth 12 --subsample 1 --colsample_bytree 1 --early_stopping_rounds 1000

# python test.py --test_path ./dataset/re_test_re.csv
