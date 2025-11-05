import torch


def parse_fasta(file_path: str):
    '''
    Parse fasta file (.fna) file into list of strings

    Parameters
    ----------
    file_path : str
        Path to the fasta sequence file

    Returns
    -------
    seq : List
        The sequence parsed so that each element is a line from the fasta file
    '''
    with open(file_path) as f:
        seq = []
        for line in f:
            line = line.rstrip()
            # do not parse the header lines
            if line.startswith('>'):
                continue
            else:
                seq.append(line)
    return seq


def get_masked_embedding(tokenizer, model, sequence: str):
    '''
    Create an embedding of a genome sequence

    Parameters
    ----------
    sequence : str
        The sequence we parsed earlier, as a single string

    Returns
    -------
    np.ndarray
        The mean embedding for the tokenized sequence
    '''
    # this splits the sequence into input:
    #  - IDs e.g. [2, 312, ... , 3671] which each correspond to a different 6-mer string in the
    # tokenizer's vocabulary
    #  - an attention mask e.g. [1, 1, 1, 0, 0] which tells the embedding model which IDs to
    # process. In our case (for now), it'll be tensor of 1s since we are including every 6-mer
    # string
    tokens = tokenizer(sequence, return_tensors="pt")
    # input_ids = tokens["input_ids"][0].tolist()
    input_ids = tokens["input_ids"]
    # print(tokenizer.convert_ids_to_tokens(input_ids))
    # print(input_ids)
    attention_mask = tokens['attention_mask']
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
    # outputs is an encoding for each *token* for each layer of the (32-layer) Transformer,
    # it has length 33 since layer 0 is an 'embedding layer'
    # to get the 'final' embeddings, take the last layer NOTE: we may not always want the last
    # layer
    embeddings = outputs.hidden_states[-1]
    attention_mask = attention_mask.unsqueeze(-1)
    masked_embeddings = embeddings * attention_mask
    mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
    return mean_embedding.squeeze().numpy()
