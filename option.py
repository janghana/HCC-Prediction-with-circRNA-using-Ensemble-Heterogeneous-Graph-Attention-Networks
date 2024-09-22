import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./dataset/train.csv', help='Path to the training data')
    parser.add_argument('--val_path', type=str, default='./dataset/val.csv', help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default='./dataset/test.csv', help='Path to the test data')
    parser.add_argument('--n_experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the saved model for inference')

    # HAN Model parameters
    parser.add_argument('--in_channels', type=int, default=2, help='Number of input channels for HAN model')
    parser.add_argument('--out_channels', type=int, default=64, help='Number of output channels for HAN model')
    parser.add_argument('--heads1', type=int, default=8, help='Number of attention heads for first HANConv layer')
    parser.add_argument('--dropout1', type=float, default=0.6, help='Dropout rate for first HANConv layer')
    parser.add_argument('--heads2', type=int, default=1, help='Number of attention heads for second HANConv layer')
    parser.add_argument('--dropout2', type=float, default=0.6, help='Dropout rate for second HANConv layer')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for HAN model')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for HAN model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train HAN model')
    
    # XGBoost parameters
    parser.add_argument('--n_estimators', type=int, default=50000, help='Number of estimators for XGBoost')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for XGBoost')
    parser.add_argument('--max_depth', type=int, default=6, help='Max depth for XGBoost')
    parser.add_argument('--subsample', type=float, default=1, help='Subsample rate for XGBoost')
    parser.add_argument('--colsample_bytree', type=float, default=1, help='Colsample bytree for XGBoost')
    parser.add_argument('--early_stopping_rounds', type=int, default=150, help='Early stopping rounds for XGBoost')

    return parser.parse_args()
