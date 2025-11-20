import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
import os

from genome_transformer_comparison.configuration import ROOT_DIR


def load_pt_files(folder_path: str):
    """
    Load all .pt tensors in a folder and return a list of tensors and their filenames.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .pt files.

    Returns
    -------
    tensors : list
        A list of loaded PyTorch tensors.
    filenames : list
        A list of filenames corresponding to the loaded tensors.
    """
    tensors = []
    filenames = []

    for fname in os.listdir(folder_path):
        if fname.endswith(".pt"):
            tensor = torch.load(os.path.join(folder_path, fname))
            tensors.append(tensor)
            filenames.append(fname)

    return tensors, filenames


def plot_cosine_similarity_heatmap(embedding_tensors, embedding_file_names, bacteria_folder):
    """
    Compute and save a cosine similarity heatmap for a set of embedding tensors.

    Parameters
    ----------
    embedding_tensors : list of torch.Tensor
        A list of embedding tensors, each with identical dimensionality.
    embedding_file_names : list of str
        Filenames corresponding to each embedding tensor. Used to generate tick labels.
    bacteria_folder : str
        Path to the bacteria-specific folder under the `outputs/` directory.
        The heatmap will be saved in a `results/` folder with the same structure.
    """
    similarity_matrix = cosine_similarity(torch.stack(embedding_tensors).numpy())
    tick_labels = ['_'.join(fname.split('_')[:2]) for fname in embedding_file_names]

    results_folder = bacteria_folder.replace('outputs', 'results')
    os.makedirs(results_folder, exist_ok=True)

    plt.figure(figsize=(32, 18))
    sns.heatmap(
        similarity_matrix,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap='viridis',
        annot=True,
        fmt=".4f"
    )
    plt.title(f"Cosine Similarity Heatmap – {os.path.basename(bacteria_folder)}")
    plt.tight_layout()

    plot_outpath = os.path.join(
        results_folder, f'cosine_similarity_{os.path.basename(bacteria_folder)}.pdf'
    )
    plt.savefig(plot_outpath)
    plt.close()

    # convert to df and save as csv
    csv_outpath = os.path.join(
        results_folder, f'cosine_similarity_{os.path.basename(bacteria_folder)}.csv'
    )
    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=tick_labels, columns=tick_labels)
    similarity_matrix_df.to_csv(csv_outpath)


def plot_pca_embeddings(embedding_tensors, embedding_file_names, bacteria_folder):
    """
    Perform PCA on embedding tensors and save a 2D scatter plot.

    Parameters
    ----------
    embedding_tensors : list of torch.Tensor
        A list of embedding tensors to be reduced via PCA. All tensors must have
        the same dimensionality.
    embedding_file_names : list of str
        Filenames corresponding to each embedding tensor. Used to label plotted points.
    bacteria_folder : str
        Path to the bacteria-specific folder under the `outputs/` directory.
        The PCA plot will be saved in a `results/` folder with the same structure.
    """

    # Stack tensors → shape (N, D)
    embeddings_matrix = torch.stack(embedding_tensors).numpy()

    # PCA → 2 components
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_matrix)

    # Labels
    tick_labels = ['_'.join(fname.split('_')[:2]) for fname in embedding_file_names]

    # Results folder
    results_folder = bacteria_folder.replace('outputs', 'results')
    os.makedirs(results_folder, exist_ok=True)

    # Plot
    plt.figure(figsize=(18, 12))
    plt.scatter(coords[:, 0], coords[:, 1])

    # Label points
    for (x, y, label) in zip(coords[:, 0], coords[:, 1], tick_labels):
        plt.text(x, y, label, fontsize=10)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Embedding Plot – {os.path.basename(bacteria_folder)}")
    plt.tight_layout()

    # Save
    outpath = os.path.join(
        results_folder,
        f'pca_embeddings_{os.path.basename(bacteria_folder)}.pdf'
    )
    plt.savefig(outpath)
    plt.close()


if __name__ == "__main__":

    cpes_or_imps = 'imps'
    bacteria_names = os.listdir(os.path.join(ROOT_DIR, 'outputs', cpes_or_imps))
    bacteria_folders = [
        os.path.join(ROOT_DIR, 'outputs', cpes_or_imps, bacteria_name)
        for bacteria_name in bacteria_names
    ]

    for bacteria_folder in bacteria_folders:
        embedding_tensors, embedding_file_names = load_pt_files(bacteria_folder)

        plot_cosine_similarity_heatmap(
            embedding_tensors, embedding_file_names, bacteria_folder)

        plot_pca_embeddings(
            embedding_tensors, embedding_file_names, bacteria_folder)
