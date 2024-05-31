import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels, filename):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 7))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=f'Label {label}', alpha=0.5)
    
    plt.legend()
    plt.title("t-SNE of Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(filename)
    plt.show()

def plot_heatmap(attention_weights, circRNA_names, gene_names, title, filename, cmap):
    plt.figure(figsize=(20, 20))
    sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap=cmap, xticklabels=gene_names, yticklabels=circRNA_names, square=True, cbar_kws={"shrink": .75})
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
