"""
Train a Colab-friendly tokenizer with configurable vocab size for a dataset.
This intentionally overwrites the dataset's default tokenizer directory so
subsequent train_t4.py runs use the newly prepared tokenizer.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import time

import rustbpe
import tiktoken
import torch

import prepare


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare tokenizer with custom vocab size.")
    parser.add_argument("--dataset", choices=prepare.DATASET_CHOICES, default="tinystorieszh")
    parser.add_argument("--vocab-size", type=int, default=prepare.VOCAB_SIZE)
    parser.add_argument("--max-chars", type=int, default=100_000_000)
    parser.add_argument("--doc-cap", type=int, default=8_000)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing tokenizer directory.")
    args = parser.parse_args()

    tokenizer_dir = prepare._tokenizer_dir(args.dataset)
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    vocab_size_path = os.path.join(tokenizer_dir, "vocab_size.txt")

    existing_vocab_size = None
    if os.path.exists(vocab_size_path):
        try:
            existing_vocab_size = int(open(vocab_size_path, "r", encoding="utf-8").read().strip())
        except Exception:
            existing_vocab_size = None

    if (
        os.path.exists(tokenizer_pkl)
        and os.path.exists(token_bytes_path)
        and existing_vocab_size == args.vocab_size
        and not args.force
    ):
        print(f"Tokenizer already exists at {tokenizer_dir} with vocab_size={args.vocab_size}")
        return 0

    if os.path.isdir(tokenizer_dir) and args.force:
        shutil.rmtree(tokenizer_dir)

    os.makedirs(tokenizer_dir, exist_ok=True)
    print(
        f"Training tokenizer: dataset={args.dataset} vocab_size={args.vocab_size:,} "
        f"max_chars={args.max_chars:,} doc_cap={args.doc_cap:,}"
    )

    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = args.vocab_size - len(prepare.SPECIAL_TOKENS)
    if vocab_size_no_special <= 0:
        raise ValueError("vocab-size must be larger than the number of special tokens.")

    tokenizer.train_from_iterator(
        prepare.text_iterator(dataset_name=args.dataset, max_chars=args.max_chars, doc_cap=args.doc_cap),
        vocab_size_no_special,
        pattern=prepare.SPLIT_PATTERN,
    )

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    token_offset = len(mergeable_ranks)
    special_tokens = {name: token_offset + i for i, name in enumerate(prepare.SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    special_set = set(prepare.SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)

    with open(os.path.join(tokenizer_dir, "dataset.txt"), "w", encoding="utf-8") as f:
        f.write(args.dataset + "\n")
    with open(vocab_size_path, "w", encoding="utf-8") as f:
        f.write(str(args.vocab_size) + "\n")

    prepare._set_active_dataset(args.dataset)
    print(f"Tokenizer ready in {time.time() - t0:.1f}s at {tokenizer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
