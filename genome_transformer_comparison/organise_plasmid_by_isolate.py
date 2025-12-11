'''
The plasmid 'assembly' files that we get are not organised by bacteria, but the file names
do come with the isolate's name in the file name. This script does some simple path operations
to organise these files so that they have the same structure as the whole genome assemblies.
'''
import os
import shutil

from genome_transformer_comparison.configuration import ROOT_DIR

cpes_or_imps = 'cpes'

plasmids_folder = os.path.join(ROOT_DIR, 'inputs', 'plasmids', cpes_or_imps)

# then the folder structure is not broken down by bacteria (i.e. plasmids)
fasta_file_names = os.listdir(plasmids_folder)

for filename in fasta_file_names:
    if not filename.endswith(".fasta"):
        continue  # skip non-fasta files

    # Extract bacterium name: first two underscore‐separated parts
    parts = filename.split("_")
    bacterium = "_".join(parts[:2])

    # Create output directory for this bacterium
    out_dir = os.path.join(plasmids_folder, bacterium)
    os.makedirs(out_dir, exist_ok=True)

    # Copy all files whose filename starts with the bacterium name
    for f in fasta_file_names:
        if f.startswith(bacterium):
            src = os.path.join(plasmids_folder, f)
            dst = os.path.join(out_dir, f)
            shutil.copy2(src, dst)
