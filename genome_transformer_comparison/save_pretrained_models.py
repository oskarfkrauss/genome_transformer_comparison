'''
Save pretrained models locally for faster loading on instantiation NOTE: run this every so often
to keep models up to date
'''
from transformers import AutoTokenizer, AutoModelForMaskedLM

from configuration import PRETRAINED_MODELS_DIR

tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    )
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    )

tokenizer.save_pretrained(PRETRAINED_MODELS_DIR)
model.save_pretrained(PRETRAINED_MODELS_DIR)
