# HCC Prediction with circRNA using Ensemble Heterogeneous Graph Attention Networks

This repository contains the implementation of a model to predict Hepatocellular Carcinoma (HCC) using circular RNA (circRNA) data. The model utilizes an Ensemble Heterogeneous Graph Attention Network (HAN) to effectively capture the relationships between circRNA and host genes and then uses XGBoost for the final prediction.

## Research Objective

The primary objective of this research is to develop a non-invasive and highly accurate method for predicting Hepatocellular Carcinoma (HCC) by leveraging the potential of circular RNAs (circRNAs) as biomarkers. Given the complex interactions between circRNAs and their host genes, we employ a Heterogeneous Graph Attention Network (HAN) to model these relationships effectively. By incorporating attention mechanisms, the HAN model identifies critical circRNAs and host genes that are indicative of HCC presence. The learned embeddings from the HAN model are subsequently used to train an XGBoost classifier, enhancing the prediction accuracy. This approach not only aims to improve the early detection and diagnosis of HCC but also provides insights into the underlying biological interactions, potentially applicable to other types of cancer diagnostics.


## Directory Structure

```
\project/
├── dataset.py
├── model.py
├── train.py
├── utils.py
└── option.py

```


## Requirements

The required Python packages can be installed using the following command:

```bash
pip install -r requirements.txt
```

### Requirements.txt

```txt
pandas==1.3.5
torch==1.10.0
scikit-learn==0.24.2
xgboost==1.5.1
torch-geometric==2.0.3
matplotlib==3.4.3
seaborn==0.11.2
```


## File Descriptions
`dataset.py`
This module contains functions to load and preprocess the data.

- load_and_preprocess_data(train_path, val_path, test_path): Loads and preprocesses the data from the provided file paths.
- create_hetero_data(features, circRNA_nodes, host_gene_nodes, labels): Creates a heterogeneous graph from the provided features, circRNA nodes, host gene nodes, and labels.

`model.py`
This module defines the HAN model and includes functions to train the XGBoost model.

- HAN: Class defining the Heterogeneous Attention Network.
- train_xgboost_model(train_embeddings, train_labels, test_embeddings, test_labels, n_estimators, learning_rate, max_depth, subsample, colsample_bytree, early_stopping_rounds): Function to train the XGBoost model.

`utils.py`
This module contains utility functions for running experiments and calculating metrics.

- train_han_model(model, train_loader, val_loader, device, optimizer, epochs): Function to train the HAN model.
- extract_embeddings(model, data, device): Function to extract embeddings from the HAN model.
- run_experiments(n_experiments, load_and_preprocess_data, create_hetero_data, HAN, train_han_model, train_xgboost_model, device, metadata, han_params, xgboost_params, train_params): Function to run multiple experiments and calculate mean and standard deviation for the results.

`option.py`
This module contains argument parser settings.

- parse_args(): Parses command line arguments.

`train.py`
This is the main script to run the entire pipeline. It uses argparse to accept various parameters for the HAN model, XGBoost model, and experiment settings, and calls the run_experiments function.

## Usage
Running the Model
To run the model, use the following command:

```bash
python train.py --train_path ./dataset/train.csv --val_path ./dataset/val.csv --test_path ./dataset/test.csv --n_experiments 5 --in_channels 2 --out_channels 64 --heads1 8 --dropout1 0.6 --heads2 1 --dropout2 0.6 --lr 0.005 --weight_decay 0.0005 --epochs 100 --n_estimators 50000 --learning_rate 0.1 --max_depth 6 --subsample 1 --colsample_bytree 1 --early_stopping_rounds 150
```

Argument Descriptions
- --train_path: Path to the training data file.
- --val_path: Path to the validation data file.
- --test_path: Path to the test data file.
- --n_experiments: Number of experiments to run.
- --in_channels: Number of input channels for HAN model.
- --out_channels: Number of output channels for HAN model.
- --heads1: Number of attention heads for the first HANConv layer.
- --dropout1: Dropout rate for the first HANConv layer.
- --heads2: Number of attention heads for the second HANConv layer.
- --dropout2: Dropout rate for the second HANConv layer.
- --lr: Learning rate for HAN model.
- --weight_decay: Weight decay for HAN model.
- --epochs: Number of epochs to train the HAN model.
- --n_estimators: Number of estimators for XGBoost.
- --learning_rate: Learning rate for XGBoost.
- --max_depth: Maximum depth for XGBoost.
- --subsample: Subsample rate for XGBoost.
- --colsample_bytree: Column sample by tree for XGBoost.
- --early_stopping_rounds: Early stopping rounds for XGBoost.


