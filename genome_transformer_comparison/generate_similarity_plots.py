'''
A script to generate within and between species similarity plots based on embeddings produced
by a Transformer model

Not the best written script of all time, sorry
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
import os

from genome_transformer_comparison.configuration import ROOT_DIR


def load_pt_files(file_paths):
    """
    Load .pt tensors from a list of file paths and return a list of tensors.

    Parameters
    ----------
    file_paths : list of str
        List of full paths to .pt files.

    Returns
    -------
    tensors : list
        A list of loaded PyTorch tensors.
    filenames : list
        Filenames extracted from the file paths.
    """
    tensors = []

    for path in file_paths:
        if path.endswith(".pt"):
            tensor = torch.load(path)
            tensors.append(tensor)

    return tensors


def plot_cosine_similarity_heatmap(
        embedding_tensors, embedding_file_names, bacteria_folder=None):
    """
    Compute and save a cosine similarity heatmap for a set of embedding tensors.

    If no bacteria folder is given we are plotting the similarity between species

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

    tick_labels = [_extract_label_for_plotting(
        fname, bacteria_folder) for fname in embedding_file_names]

    plt.figure(figsize=(32, 18))
    sns.heatmap(
        similarity_matrix,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap='viridis',
        annot=False,
        fmt=".4f"
    )

    if bacteria_folder is not None:
        plt.title(f"Cosine Similarity Heatmap - {os.path.basename(bacteria_folder)}")
        plt.tight_layout()

        results_folder = bacteria_folder.replace('outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        plot_outpath = os.path.join(
            results_folder, f'cosine_similarity_{os.path.basename(bacteria_folder)}.pdf'
        )
        # convert to df and save as csv
        csv_outpath = os.path.join(
            results_folder, f'cosine_similarity_{os.path.basename(bacteria_folder)}.csv')
    else:
        results_folder = os.path.join(ROOT_DIR, 'results', 'whole_genomes', cpes_or_imps)
        os.makedirs(results_folder, exist_ok=True)

        plot_outpath = os.path.join(results_folder, 'cosine_similarity.pdf')
        # convert to df and save as csv
        csv_outpath = os.path.join(results_folder, 'cosine_similarity.csv')

    plt.savefig(plot_outpath)
    plt.close()

    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=tick_labels, columns=tick_labels)
    similarity_matrix_df.to_csv(csv_outpath)


def plot_pca_embeddings(
        embedding_tensors, embedding_file_names, bacteria_folder=None):
    """
    Perform PCA on embedding tensors and save a 2D scatter plot.

    Apoliges in advance, this is an awful funciton.

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

    # PCA → 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_matrix)

    # Extract labels from file paths
    labels = [_extract_label_for_plotting(fname, bacteria_folder) for fname in embedding_file_names]

    if bacteria_folder is not None:
        # All points one colour
        colours = ["C0"] * len(labels)

        # Results folder
        results_folder = bacteria_folder.replace('outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        # Plot
        plt.figure(figsize=(18, 12))
        plt.scatter(coords[:, 0], coords[:, 1])

        # Label points
        for (x, y, label) in zip(coords[:, 0], coords[:, 1], labels):
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

    else:
        # Unique labels → assign colors, make sure we get just the bacteria name from the label,
        # the extract_label_for_plotting function returns bacteria_assembly_*
        unique_bacteria = sorted(set(['_'.join(lbl.split('_')[:2]) for lbl in labels]))
        label_to_colour = {label: plt.cm.tab10(i % 10) for i, label in enumerate(unique_bacteria)}
        colours = [label_to_colour[x] for x in ['_'.join(lbl.split('_')[:2]) for lbl in labels]]
        # Plot
        plt.figure(figsize=(14, 10))
        plt.scatter(coords[:, 0], coords[:, 1], c=colours, s=80)

        # Axis labels
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

        # Add a legend showing label -> color
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=label_to_colour[lab])
                   for lab in unique_bacteria]
        plt.legend(handles, unique_bacteria, title="Labels", fontsize=10)

        results_folder = os.path.join(ROOT_DIR, 'results', 'whole_genomes', cpes_or_imps)
        os.makedirs(results_folder, exist_ok=True)

        outpath = os.path.join(results_folder, 'pca_embeddings.pdf')

    plt.savefig(outpath)
    plt.close()


def _extract_label_for_plotting(file_path, bacteria_folder):
    """
    Extract a label from a file path.

    Basically, some really bad string formatting to get the right labels on things,
    if we're looking at within species, we require the name of the assembly file.

    Parameters
    ----------
    path : str
        Full path to the embedding file.

    bacteria_folder : str or None
        If not None, only the filename is used to form the label.
        If None, the label is formed using the last directory plus
        the filename.

    Returns
    -------
    label : str
        A label constructed by splitting the selected path component(s)
        on underscores and keeping the first two segments.
    """
    parts = os.path.normpath(file_path).split(os.sep)

    if bacteria_folder is not None:
        # Use only the filename
        target = parts[-1]
    else:
        # Use last folder + filename
        target = "_".join(parts[-2:])

    return "_".join(target.split("_")[:2])


if __name__ == "__main__":

    cpes_or_imps = 'imps'
    bacteria_names = os.listdir(
        os.path.join(ROOT_DIR, 'outputs', 'whole_genomes', cpes_or_imps))

    all_tensor_file_paths = []

    for bacteria_name in bacteria_names:
        bacteria_folder = os.path.join(ROOT_DIR, 'outputs', 'whole_genomes',
                                       cpes_or_imps, bacteria_name)

        bacteria_tensor_file_names = os.listdir(
            os.path.join(ROOT_DIR, 'outputs', 'whole_genomes', cpes_or_imps, bacteria_name))

        bacteria_tensor_file_paths = [os.path.join(
            ROOT_DIR, 'outputs', 'whole_genomes', cpes_or_imps, bacteria_name, file_name)
            for file_name in bacteria_tensor_file_names]

        all_tensor_file_paths.extend(bacteria_tensor_file_paths)

        # get plots for within species similarity
        embedding_tensors = load_pt_files(bacteria_tensor_file_paths)

        plot_cosine_similarity_heatmap(
            embedding_tensors, bacteria_tensor_file_paths, bacteria_folder)
        plot_pca_embeddings(
            embedding_tensors, bacteria_tensor_file_paths, bacteria_folder)

    # plot the similarities and pca for all embedding, i.e. asses between species similarity
    all_embedding_tensors = load_pt_files(all_tensor_file_paths)
    plot_cosine_similarity_heatmap(
        all_embedding_tensors, all_tensor_file_paths)
    plot_pca_embeddings(
        all_embedding_tensors, all_tensor_file_paths)
