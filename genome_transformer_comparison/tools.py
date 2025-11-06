import torch


def parse_fasta(file_path: str):
    '''
    Parse fasta file (.fna or .fasta) file into a single string

    Parameters
    ----------
    file_path : str
        Path to the fasta assembly file

    Returns
    -------
    seq : str
        The sequence parsed into a single string
    '''
    with open(file_path) as f:
        seq = ''
        for line in f:
            line = line.rstrip()
            # ignore lines containing read headers
            if line.startswith('>'):
                continue
            else:
                seq = seq + line
    return seq


def split_sequence_for_tokenizer(
        sequence: str, max_length: int, overlap: int = 0) -> list:
    """
    Split a long genome sequence string into a list of substrings each no longer than
    max_length, optionally with overlap between consecutive chunks.

    Parameters
    ----------
    sequence : str
        Raw sequence (may contain newlines). This will be normalized to uppercase.
    max_length : int
        Maximum length (in characters) of each chunk. Choose this to match the tokenizer's
        maximum input size (or slightly smaller).
    overlap : int
        Number of characters to overlap between consecutive chunks (0 = no overlap).

    Returns
    -------
    List[str]
        List of sequence chunks suitable for passing individually to the tokenizer.
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_length:
        raise ValueError("overlap must be smaller than max_length")

    # normalize sequence (not sure this is necessary)
    seq = sequence.replace("\n", "").replace("\r", "").upper()

    chunks = []
    step = max_length - overlap
    start = 0
    seq_len = len(seq)
    while start < seq_len:
        end = start + max_length
        chunks.append(seq[start:end])
        start += step
    return chunks


def get_chunk_embedding(tokenizer, model, sequence: str, device=None):
    """
    Create an embedding of a 'chunk' of a genome sequence (on GPU if available).

    Parameters
    ----------
    sequence : str
        A 'chunk' of the genome sequence.
    device : torch.device or None
        Device to run the model on (CPU or GPU). Defaults to CPU.

    Returns
    -------
    torch.Tensor
        The embedding for the tokenized chunk (shape [seq_len, hidden_dim])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize and move to device
    tokens = tokenizer(sequence, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens['attention_mask']

    model = model.to(device)
    model.eval()  # ensure evaluation mode

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)

    # Get last hidden state (remove batch dimension)
    embeddings = outputs.hidden_states[-1].squeeze(0)
    return embeddings.cpu()  # move back to CPU for stacking/mean
