import requests
import jsonlines
import zstandard
from transformer_lens.utils import get_dataset
from utils import get_model_family
import argparse
import os
import io
import math
import datasets
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from datasets import Dataset

DATASET_ALIASES = {
    "openwebtext": "stas/openwebtext-10k",
    "owt": "stas/openwebtext-10k",
    "pile": "NeelNanda/pile-10k",
    "c4": "NeelNanda/c4-10k",
    "code": "NeelNanda/code-10k",
    "python": "NeelNanda/code-10k",
    "c4_code": "NeelNanda/c4-code-20k",
    "c4-code": "NeelNanda/c4-code-20k",
    "wiki": "NeelNanda/wiki-10k",
}

PILE_SUBSET_ALIASES = {
    'ArXiv': 'arxiv',
    'BookCorpus2': 'bookcorpus2',
    'Books3': 'books3',
    'DM Mathematics': 'dm_mathematics',
    'Enron Emails': 'enron_emails',
    'EuroParl': 'europarl',
    'FreeLaw': 'freelaw',
    'Github': 'github',
    'Gutenberg (PG-19)': 'gutenberg',
    'HackerNews': 'hackernews',
    'NIH ExPorter': 'nih_exporter',
    'OpenSubtitles': 'opensubtitles',
    'OpenWebText2': 'openwebtext2',
    'PhilPapers': 'philpapers',
    'Pile-CC': 'pile_cc',
    'PubMed Abstracts': 'pubmed_abstracts',
    'PubMed Central': 'pubmed_central',
    'StackExchange': 'stackexchange',
    'USPTO Backgrounds': 'uspto_backgrounds',
    'Ubuntu IRC': 'ubuntu_irc',
    'Wikipedia (en)': 'wikipedia',
    'YoutubeSubtitles': 'youtubesubtitles'
}


def get_pile_split(split='test'):
    PILE_URL = f'https://the-eye.eu/public/AI/pile/{split}.jsonl.zst'
    response = requests.get(PILE_URL, stream=True)
    response.raise_for_status()

    dctx = zstandard.ZstdDecompressor()
    stream_reader = dctx.stream_reader(io.BytesIO(response.content))
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

    lines = []
    with jsonlines.Reader(text_stream) as reader:
        for obj in reader:
            lines.append({
                'text': obj['text'],
                'subset': obj['meta']['pile_set_name']
            })
    ds = datasets.Dataset.from_list(lines)
    return ds


def tokenize_pile_subsets(pile_ds, model, ctx_len=512):
    seq_char_len = np.array([len(t) for t in pile_ds['text']])
    valid_ixs = np.arange(len(pile_ds))[seq_char_len > 50]
    ds = pile_ds.select(valid_ixs)

    seq_subset = np.array(ds['subset'])
    subsets = np.unique(seq_subset)

    sub_ds_dict = {}
    print(subsets)
    for subset in subsets:
        print('Tokenizing subset:', subset)
        mask = seq_subset == subset
        sub_ds = ds.select(np.arange(len(ds))[mask])
        sub_ds_tokens = tokenize_and_concatenate(
            sub_ds, model.tokenizer, max_length=ctx_len)

        sub_ds_dict[subset] = sub_ds_tokens

    return sub_ds_dict


def create_pile_subset(model_family, n_tokens, n_tokens_name):
    base_path = os.path.join('token_datasets', model_family)
    dsets = []
    for ds_file in os.listdir(base_path):
        if 'all' in ds_file:
            continue
        ds = datasets.load_from_disk(os.path.join(base_path, ds_file))
        parent_ds, split, subset, ctx_len = ds_file.split('.')
        ds = ds.add_column('subset', [subset for _ in range(len(ds))])
        dsets.append(ds)

    all_ds = datasets.concatenate_datasets(dsets)
    ctx_len = len(all_ds[0]['tokens'])
    n_sequences = math.ceil(n_tokens / ctx_len)
    subsample_ds = all_ds.shuffle().select(range(n_sequences))

    save_name = f'pile.test.all-{n_tokens_name}.{ctx_len}'
    save_path = os.path.join(base_path, save_name)
    subsample_ds.save_to_disk(save_path)


def chunk_streaming_with_hf_tokenizer(streaming_ds, hf_tokenizer, ctx_len, total_token_budget):
    """
    Generator that:
      1. tokenizes each example’s text on-the-fly,
      2. appends to a small buffer,
      3. yields exactly floor(total_token_budget/ctx_len) full-length chunks,
         then returns (never processes more text).
    """
    max_chunks = total_token_budget // ctx_len
    emitted = 0
    buffer = []

    for ex in streaming_ds:
        # Default add_special_tokens=True → each example gets its EOS/BOS if model demands
        ids = hf_tokenizer(
            ex["text"],
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True
        )["input_ids"]
        buffer.extend(ids)

        while len(buffer) >= ctx_len and emitted < max_chunks:
            chunk = buffer[:ctx_len]
            yield {"tokens": chunk}
            emitted += 1
            buffer = buffer[ctx_len:]
            if emitted >= max_chunks:
                return

    # If the dataset runs out before we hit max_chunks, we just stop.
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--hf_dataset', default='EleutherAI/pile', help='Name of HuggingFace dataset')
    parser.add_argument(
        '--hf_dataset_split', default='test')
    parser.add_argument(
        '--ctx_len', default=512, type=int, help='Context length')
    parser.add_argument(
        '--n_seq', default=-1, type=int, help='Number of sequences (unused for token‐limit path)')
    parser.add_argument(
        '--total_tokens', default=100000000, type=int,
        help='Total token budget (e.g. 100_000_000).')
    parser.add_argument(
        '--output_dir', default='token_datasets', help='Path to save dataset')
    parser.add_argument(
        '--use_hf_fast_tokenizer', action='store_true',
        help='If set, replace transformer_lens tokenizer with HF tokenizer.')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device='cpu')
    model_family = get_model_family(args.model)

    # If the user specified --use_hf_fast_tokenizer, load a HF Fast tokenizer:
    hf_tokenizer = None
    if args.use_hf_fast_tokenizer:
        # We match the model name to HF. E.g. "pythia-70m" → "EleutherAI/pythia-70m"
        hf_name = f"EleutherAI/{args.model}"
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)

    # —— get data ——
    if args.hf_dataset in DATASET_ALIASES:
        dataset = get_dataset(args.hf_dataset)
    elif args.hf_dataset == 'EleutherAI/pile':
        print('Downloading pile (custom zst loader)…')
        dataset = get_pile_split(args.hf_dataset_split)
    else:
        dataset = datasets.load_dataset(
            args.hf_dataset, split=args.hf_dataset_split, streaming=True)

    # —— tokenize and save ——
    if args.hf_dataset == 'EleutherAI/pile':
        # (unchanged: you still want to tokenize each subset separately)
        ds_dict = tokenize_pile_subsets(dataset, model, ctx_len=args.ctx_len)
        for subset, sub_ds in ds_dict.items():
            subset_name = PILE_SUBSET_ALIASES[subset]
            save_path = os.path.join(
                args.output_dir, model_family,
                f'pile.{args.hf_dataset_split}.{subset_name}.{args.ctx_len}'
            )
            os.makedirs(save_path, exist_ok=True)
            sub_ds.save_to_disk(save_path)

    else:
        # If user wants to simply select N sequences, they can still use n_seq:
        if args.n_seq > 0 and args.total_tokens < 0:
            dataset = dataset.select(range(args.n_seq))

        # If user passed --use_hf_fast_tokenizer AND total_tokens > 0,
        # we do a _streaming token‐limit_ path. Otherwise we fall back to
        # the original tokenize_and_concatenate.
        if args.use_hf_fast_tokenizer and args.total_tokens > 0:
            # 1) If the dataset object is not already streaming, force it to be:
            if not getattr(dataset, "streaming", False):
                # Convert a non‐streaming Dataset into an in‐memory list, then
                # re‐wrap as a streaming Iterable—BUT for large datasets that
                # will load everything into RAM. If you know the HF split is
                # already streaming-capable, skip this.
                dataset = datasets.load_dataset(
                    args.hf_dataset, split=args.hf_dataset_split, streaming=True)

            # 2) Filter out very short texts (like before)
            dataset = dataset.filter(lambda x: len(x["text"]) > 50)

            # 3) Build the generator of exactly (total_tokens // ctx_len) chunks
            gen = chunk_streaming_with_hf_tokenizer(
                streaming_ds=dataset,
                hf_tokenizer=hf_tokenizer,
                ctx_len=args.ctx_len,
                total_token_budget=args.total_tokens
            )

            # 4) Materialize into a list of dicts (each has key “tokens” → List[int] of length ctx_len)
            chunks = list(gen)
            limited_ds = Dataset.from_list(chunks)
            limited_ds = limited_ds.with_format("torch", columns=["tokens"])

            # 5) Save to disk
            millions = args.total_tokens // 1_000_000
            save_dir = os.path.join(
                args.output_dir,
                model_family,
                f'{args.hf_dataset_split}.streamed-{millions}M.{args.ctx_len}'
            )
            os.makedirs(save_dir, exist_ok=True)
            limited_ds.save_to_disk(save_dir)

            print(f"✅ Saved {len(limited_ds)} chunks (~{len(limited_ds)*args.ctx_len} tokens) to {save_dir}")

        else:
            # fallback: original “select n_seq, then tokenize_and_concatenate”
            if args.n_seq > 0:
                dataset = dataset.select(range(args.n_seq))

            token_dataset = tokenize_and_concatenate(
                dataset, model.tokenizer, max_length=args.ctx_len)

            save_path = os.path.join(
                args.output_dir, model_family,
                f'{args.hf_dataset}.{args.n_seq}.{args.ctx_len}'
            )
            os.makedirs(save_path, exist_ok=True)
            token_dataset.save_to_disk(save_path)

            print(f"✅ Saved tokenize_and_concatenate output to {save_path}")
