'''
A script to generate within and between species similarity plots based on embeddings produced
by a Transformer model
'''
import os

from genome_transformer_comparison.configuration import ROOT_DIR
from genome_transformer_comparison.plotting_tools import (
    load_pt_files, plot_cosine_similarity_heatmap, plot_euclidean_distance_heatmap,
    plot_pca_embeddings, plot_umap_embeddings
)

if __name__ == "__main__":

    transformer_model = 'NucleotideTransformer_2.5B'
    whole_genomes_or_plasmids = 'whole_genomes'
    cpes_or_imps = 'imps'
    base_dir = os.path.join(
        ROOT_DIR, 'outputs', transformer_model, whole_genomes_or_plasmids, cpes_or_imps)
    bacteria_names = os.listdir(base_dir)

    all_tensor_file_paths = []

    # get within species simiality plots and metrics
    for bacteria_name in bacteria_names:
        within_species = True
        bacteria_folder = os.path.join(base_dir, bacteria_name)

        bacteria_tensor_file_paths = [
            os.path.join(bacteria_folder, file_name) for file_name in os.listdir(bacteria_folder)
            ]

        all_tensor_file_paths.extend(bacteria_tensor_file_paths)

        # get plots for within species similarity
        embedding_tensors = load_pt_files(bacteria_tensor_file_paths)

        plot_cosine_similarity_heatmap(
            embedding_tensors, bacteria_tensor_file_paths, within_species)
        plot_euclidean_distance_heatmap(
            embedding_tensors, bacteria_tensor_file_paths, within_species)
        plot_pca_embeddings(
            embedding_tensors, bacteria_tensor_file_paths, within_species)
        plot_umap_embeddings(
            embedding_tensors, bacteria_tensor_file_paths, within_species)

    within_species = False
    # plot the similarities and pca for all embedding, i.e. asses between species similarity
    all_embedding_tensors = load_pt_files(all_tensor_file_paths)
    plot_cosine_similarity_heatmap(
        all_embedding_tensors, all_tensor_file_paths)
    plot_euclidean_distance_heatmap(
        all_embedding_tensors, all_tensor_file_paths)
    plot_pca_embeddings(
        all_embedding_tensors, all_tensor_file_paths)
    plot_umap_embeddings(
        all_embedding_tensors, all_tensor_file_paths)
