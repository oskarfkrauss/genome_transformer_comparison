'''
Save pretrained models locally for faster loading on instantiation NOTE: run this every so often
to keep models up to date
'''
import os
from transformers import AutoTokenizer, AutoModel

from configuration import PRETRAINED_MODELS_DIR


def load_and_save_model(model_name: str, model_dir: str):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Save model & tokenizer
    tokenizer.save_pretrained(os.path.join(PRETRAINED_MODELS_DIR, model_dir, "tokenizer"))
    model.save_pretrained(os.path.join(PRETRAINED_MODELS_DIR, model_dir, "model"))


# 1. Nucleotide model 2.5B
load_and_save_model(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species", "NucleotideTransformer_2.5B")

# 2. DNABERT-S
load_and_save_model("zhihan1996/DNABERT-S", "DNABERT_S")

# 3. HyenaDNA
load_and_save_model("LongSafari/hyenadna-medium-160k-seqlen-hf", "HyenaDNA_medium_160k")

# 4. GENA-LM
load_and_save_model("AIRI-Institute/gena-lm-bert-base", "GENA_LM_base")
