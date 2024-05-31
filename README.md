# HCC Prediction with circRNA using Ensemble Heterogeneous Graph AttentionNetworks

This repository contains code for predicting hepatocellular carcinoma (HCC) using circRNA and host gene data with Heterogeneous Graph Attention Networks (HAN). The project includes data preprocessing, model training, and visualization.

## Project Structure

├── dataset.py # Dataset processing module

├── models.py # Model definition module

├── train.py # Training and evaluation module

├── visualization.py # Visualization functions module

├── main.py # Main script to run the entire process

├── requirements.txt # Python dependencies

└── README.md # Project documentation


## Requirements
- Python 3.7+
- Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Place your dataset files (`train.csv`, `val.csv`, `test.csv`) in the `./dataset/` directory or specify their paths when running the script.

### Training and Evaluation

Run the main script with the necessary arguments:
```bash
python3 main.py --train_path ./dataset/re_train.csv --val_path ./dataset/re_val.csv --test_path ./dataset/re_test.csv --xgb_model_path xgboost_model_5.json --hidden_channels 512 --out_channels 2 --num_heads 8 --num_layers 2
```

### Arguments
--train_path: Path to the training dataset (default: ./dataset/re_train.csv)
--val_path: Path to the validation dataset (default: ./dataset/re_val.csv)
--test_path: Path to the test dataset (default: ./dataset/re_test.csv)
--xgb_model_path: Path to the pre-trained XGBoost model (default: xgboost_model_5.json)
--hidden_channels: Number of hidden channels in the HAN model (default: 512)
--out_channels: Number of output channels in the HAN model (default: 2)
--num_heads: Number of attention heads in the HAN model (default: 8)
--num_layers: Number of layers in the HAN model (default: 2)

### File Descriptions
dataset.py: Contains the CircRNADataset class for loading and processing circRNA data.
models.py: Defines the CircHANConv and HANModel classes for the HAN model.
train.py: Contains functions for training the model (train_model), evaluating metrics (evaluate_metrics), and extracting embeddings and attention weights (get_embeddings_and_attention).
visualization.py: Provides functions for visualizing the model's outputs (plot_tsne and plot_heatmap).
main.py: The main script to run the entire process, from data loading to model training and evaluation.
Results
The project aims to use the HAN model for predicting HCC with high accuracy by leveraging attention mechanisms to understand the relationships between circRNA and host genes. Visualization techniques like t-SNE and heatmaps are used to analyze and present the model's performance and insights.

### Expected Outcomes
This research introduces a novel approach to HCC prediction using Ensemble Heterogeneous Graph Attention Networks (HAN) with circRNA data, providing a non-invasive and cost-effective diagnostic tool with high accuracy. By leveraging the attention mechanisms within HAN, we gain a deeper understanding of the interactions between circRNA and their host genes. This approach not only enhances the accuracy of HCC predictions but also identifies key circRNAs and genes associated with HCC, offering valuable insights for further biological research and potential therapeutic targets. This methodology, while focused on HCC, has the potential to be applied to other types of cancer, broadening its impact in the medical field. Ultimately, the implementation of this model can improve early diagnosis and treatment strategies for HCC, contributing to better patient outcomes and advancing the field of cancer research.


