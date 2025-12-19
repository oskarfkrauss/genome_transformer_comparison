import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import torch
import umap

from genome_transformer_comparison.configuration import ROOT_DIR


def get_pt_files_dict(transformer_model, whole_genomes_or_plasmids, cpes_or_imps):
    """
    Collect paths to embedding tensor files and organize them by bacteria.

    This function scans the directory:
        ROOT_DIR/outputs/<transformer_model>/<whole_genomes_or_plasmids>/<cpes_or_imps>/

    Each subdirectory is assumed to correspond to a bacterial species and
    contains embedding tensor files (e.g. `.pt`). For each bacterial species,
    the full file paths are collected and stored in a dictionary.

    An additional key, 'all', contains a flat list of all tensor file paths
    across all bacterial species.

    Parameters
    ----------
    transformer_model : str
        Name of the transformer model directory (e.g. 'NucleotideTransformer_2.5B').
    whole_genomes_or_plasmids : str
        Dataset type (e.g. 'whole_genomes' or 'plasmids').
    cpes_or_imps : str
        Dataset category (e.g. 'cpes' or 'imps').

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping bacterial species names to lists of full paths
        to embedding tensor files. The special key 'all' maps to a list
        containing all tensor file paths across all species.

    Notes
    -----
    - The function does not filter files by extension.
    - The order of files and bacteria depends on the filesystem order.
    - No validation is performed to ensure directories or files exist.
    """
    base_dir = os.path.join(
        ROOT_DIR, 'outputs', transformer_model, whole_genomes_or_plasmids, cpes_or_imps)
    bacteria_names = os.listdir(base_dir)

    all_tensor_file_paths = []
    pt_files_dict = {'all': []}

    for bacteria_name in bacteria_names:
        bacteria_folder = os.path.join(base_dir, bacteria_name)
        bacteria_tensor_file_paths = [
            os.path.join(bacteria_folder, file_name) for file_name in os.listdir(bacteria_folder)
            ]
        pt_files_dict[bacteria_name] = bacteria_tensor_file_paths
        all_tensor_file_paths.extend(bacteria_tensor_file_paths)

    pt_files_dict['all'] = all_tensor_file_paths

    return pt_files_dict


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
    """
    tensors = []

    for path in file_paths:
        if path.endswith(".pt"):
            tensor = torch.load(path)
            tensors.append(tensor)

    return tensors


def plot_cosine_similarity_heatmap(
        embedding_tensors, embedding_file_names, within_species=False):
    """
    Compute and save a cosine similarity heatmap for a set of embedding tensors.

    If no bacteria folder is given we are plotting the similarity between species

    Parameters
    ----------
    embedding_tensors : list of torch.Tensor
        A list of embedding tensors, each with identical dimensionality.
    embedding_file_names : list of str
        Filenames corresponding to each embedding tensor. Used to generate tick labels.
    within_species : bool
        Whether we are plotting within species similarity, defaults to False,
        across species similarity
    """
    similarity_matrix = cosine_similarity(torch.stack(embedding_tensors).numpy())

    tick_labels = [_extract_label_for_plotting(
        fname, within_species) for fname in embedding_file_names]

    plt.figure(figsize=(32, 18))
    sns.heatmap(
        similarity_matrix,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap='viridis',
        annot=False,
        fmt=".4f"
    )

    if within_species:
        bacteria_folder = os.path.dirname(embedding_file_names[0])
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
        plt.title("Cosine Similarity Heatmap - Across Species")
        plt.tight_layout()
        results_folder = os.path.dirname(os.path.dirname(embedding_file_names[0])).replace(
            'outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        plot_outpath = os.path.join(results_folder, 'cosine_similarity.pdf')
        # convert to df and save as csv
        csv_outpath = os.path.join(results_folder, 'cosine_similarity.csv')

    plt.savefig(plot_outpath)
    plt.close()

    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=tick_labels, columns=tick_labels)
    similarity_matrix_df.to_csv(csv_outpath)

    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=tick_labels, columns=tick_labels)
    similarity_matrix_df.to_csv(csv_outpath)


def plot_euclidean_distance_heatmap(
        embedding_tensors, embedding_file_names, within_species=False):
    """
    Compute and save a euclidean distance heatmap for a set of embedding tensors.

    If no bacteria folder is given we are plotting the similarity between species

    Parameters
    ----------
    embedding_tensors : list of torch.Tensor
        A list of embedding tensors, each with identical dimensionality.
    embedding_file_names : list of str
        Filenames corresponding to each embedding tensor. Used to generate tick labels.
    within_species : bool
        Whether we are plotting within species similarity, defaults to False,
        across species similarity
    """
    similarity_matrix = euclidean_distances(torch.stack(embedding_tensors).numpy())

    tick_labels = [_extract_label_for_plotting(
        fname, within_species) for fname in embedding_file_names]

    plt.figure(figsize=(32, 18))
    sns.heatmap(
        similarity_matrix,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap='viridis',
        annot=False,
        fmt=".4f"
    )

    if within_species:
        bacteria_folder = os.path.dirname(embedding_file_names[0])
        plt.title(f"Euclidean Distance Heatmap - {os.path.basename(bacteria_folder)}")
        plt.tight_layout()

        results_folder = bacteria_folder.replace('outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        plot_outpath = os.path.join(
            results_folder, f'euclidean_distance_{os.path.basename(bacteria_folder)}.pdf'
        )
        # convert to df and save as csv
        csv_outpath = os.path.join(
            results_folder, f'euclidean_distance_{os.path.basename(bacteria_folder)}.csv')
    else:
        plt.title("Euclidean Distance Heatmap - Across Species")
        plt.tight_layout()
        results_folder = os.path.dirname(os.path.dirname(embedding_file_names[0])).replace(
            'outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        plot_outpath = os.path.join(results_folder, 'euclidean_distance.pdf')
        # convert to df and save as csv
        csv_outpath = os.path.join(results_folder, 'euclidean_distance.csv')

    plt.savefig(plot_outpath)
    plt.close()

    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=tick_labels, columns=tick_labels)
    similarity_matrix_df.to_csv(csv_outpath)


def plot_pca_embeddings(
        embedding_tensors, embedding_file_names, within_species=False):
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
    within_species : bool
        Whether we are plotting within species similarity, defaults to False,
        across species similarity
    """
    # Stack tensors → shape (N, D)
    embeddings_matrix = torch.stack(embedding_tensors).numpy()

    # PCA → 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_matrix)

    # Extract labels from file paths
    labels = [_extract_label_for_plotting(fname, within_species) for fname in embedding_file_names]

    if within_species:
        bacteria_folder = os.path.dirname(embedding_file_names[0])
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
        plt.title(f"PCA Embedding Plot - {os.path.basename(bacteria_folder)}")
        plt.tight_layout()

        # Save
        outpath = os.path.join(
            results_folder,
            f'pca_embeddings_{os.path.basename(bacteria_folder)}.pdf'
        )

    else:
        # Assign colors, make sure we get just the bacteria name from the label,
        # the extract_label_for_plotting function returns bacteria_assembly_*
        bacteria_name_in_label = ['_'.join(lbl.split('_')[:2]) for lbl in labels]
        unique_bacteria = sorted(set(bacteria_name_in_label))
        label_to_colour = {label: plt.cm.tab10(i % 10) for i, label in enumerate(unique_bacteria)}
        colours = [label_to_colour[x] for x in bacteria_name_in_label]
        # Plot
        plt.figure(figsize=(14, 10))
        plt.scatter(coords[:, 0], coords[:, 1], c=colours, s=80)

        # Axis labels
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("PCA Embedding Plot - Across Species")
        plt.tight_layout()

        # Add a legend showing label -> color
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=label_to_colour[lab])
                   for lab in unique_bacteria]
        plt.legend(handles, unique_bacteria, title="Labels", fontsize=10)

        results_folder = os.path.dirname(os.path.dirname(embedding_file_names[0])).replace(
            'outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        outpath = os.path.join(results_folder, 'pca_embeddings.pdf')

    plt.savefig(outpath)
    plt.close()


def plot_umap_embeddings(
        embedding_tensors, embedding_file_names, within_species=False,
        n_neighbors=10, min_dist=0.1):
    """
    Reduce embeddings to 2D using UMAP and save a scatter plot.

    Parameters
    ----------
    embedding_tensors : list of torch.Tensor or numpy arrays
        A list of embedding tensors/arrays to be reduced via UMAP. All must have
        the same dimensionality.
    embedding_file_names : list of str
        Filenames corresponding to each embedding tensor. Used to label plotted points.
    within_species : bool
        Whether we are plotting within species similarity, defaults to False,
        across species similarity
    n_neighbors : int
        UMAP n_neighbors parameter (controls local vs global structure).
    min_dist : float
        UMAP min_dist parameter (controls how tightly UMAP packs points).
    random_state : int
        Random seed for reproducibility.
    """
    embeddings_matrix = torch.stack(embedding_tensors).cpu().numpy()

    # UMAP → 2D
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_jobs=1)
    coords = reducer.fit_transform(embeddings_matrix)

    # Extract labels from file paths
    labels = [_extract_label_for_plotting(fname, within_species) for fname in embedding_file_names]

    if within_species:
        bacteria_folder = os.path.dirname(embedding_file_names[0])
        # All points one colour
        colours = ["C0"] * len(labels)

        # Results folder (mirror outputs -> results)
        results_folder = bacteria_folder.replace('outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        # Plot
        plt.figure(figsize=(18, 12))
        plt.scatter(coords[:, 0], coords[:, 1], c=colours)

        # Label points
        for (x, y, label) in zip(coords[:, 0], coords[:, 1], labels):
            plt.text(x, y, label, fontsize=10)

        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.title(f"UMAP Embedding Plot - {os.path.basename(bacteria_folder)}")
        plt.tight_layout()

        outpath = os.path.join(
            results_folder,
            f'umap_embeddings_{os.path.basename(bacteria_folder)}.pdf'
        )

    else:
        # assign colors. Keep same bacteria_name extraction logic as before.
        bacteria_name_in_label = ['_'.join(lbl.split('_')[:2]) for lbl in labels]
        unique_bacteria = sorted(set(bacteria_name_in_label))
        label_to_colour = {label: plt.cm.tab10(i % 10) for i, label in enumerate(unique_bacteria)}
        colours = [label_to_colour[x] for x in bacteria_name_in_label]

        plt.figure(figsize=(14, 10))
        plt.scatter(coords[:, 0], coords[:, 1], c=colours, s=80)

        # Axis labels
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.title("UMAP Embedding Plot - Across Species")
        plt.tight_layout()

        # Add a legend showing label -> color
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=label_to_colour[lab])
                   for lab in unique_bacteria]
        plt.legend(handles, unique_bacteria, title="Labels", fontsize=10)

        # Derive results folder from first file path (similar logic to your original)
        results_folder = os.path.dirname(os.path.dirname(embedding_file_names[0])).replace(
            'outputs', 'results')
        os.makedirs(results_folder, exist_ok=True)

        outpath = os.path.join(results_folder, 'umap_embeddings.pdf')

    plt.savefig(outpath)
    plt.close()


def _extract_label_for_plotting(file_path, within_species=False):
    """
    Extract a label from a file path.

    Basically, some really bad string formatting to get the right labels on things,
    if we're looking at within species, we require the name of the assembly file.

    Parameters
    ----------
    path : str
        Full path to the embedding file.
    within_species : bool
        Determines how much of the file path to use when labelling points, if we are looking
        at across species similarity.

    Returns
    -------
    label : str
        A label constructed by splitting the selected path component(s)
        on underscores and keeping the first two segments.
    """
    parts = os.path.normpath(file_path).split(os.sep)

    if within_species:
        # Use only the filename
        target = parts[-1]
    else:
        # Use last folder + filename
        target = "_".join(parts[-2:])

    return "_".join(target.split("_")[:2])
