'''
A script to generate within and between species similarity plots based on embeddings produced
by a Transformer model
'''
from genome_transformer_comparison.plotting_tools import (
    get_pt_files_dict, load_pt_files, plot_cosine_similarity_heatmap,
    plot_euclidean_distance_heatmap, plot_pca_embeddings, plot_umap_embeddings
)

if __name__ == "__main__":

    transformer_model = 'NucleotideTransformer_2.5B'
    whole_genomes_or_plasmids = 'whole_genomes'
    cpes_or_imps = 'cpes'
    pt_files_dict = get_pt_files_dict(
        transformer_model, whole_genomes_or_plasmids, cpes_or_imps)

    for bacteria_name in pt_files_dict.keys():
        within_species = False if bacteria_name == 'all' else True

        bacteria_tensor_file_paths = pt_files_dict[bacteria_name]

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
