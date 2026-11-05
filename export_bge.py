# Export bge-base-en-v1.5 for embed_mpi.c. Run it in Google Colab, in one cell:
#
#   !pip install -q datasets
#   %run export_bge.py
#
# It makes bge_data.tar, with these files:
#   bge.bin        the model weights, as float32
#   vocab.txt      the tokenizer's vocabulary
#   texts.txt      1000 news texts to search through
#   check.txt      a few texts to test the C code with
#   check.bin      their vectors, computed here in Python

import struct
import tarfile

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-base-en-v1.5"
MAGIC_CLS = 0x4d424543
N_TEXTS = 1000

CHECK_TEXTS = [
    "The cat sits on the mat.",
    "Stocks rallied on Wall Street after the earnings report!",
    "A very long sentence, with commas; semicolons: and (brackets), to test the tokenizer.",
    "Crème brûlée in São Paulo costs $5.50 - it's naïve.",
    "Cluj-Napoca este un oraș din România, în Transilvania.",
    "supercalifragilisticexpialidocious xyzzy",
]


def export_model(model, path):
    """Write the weights in the order load_model() in embed_mpi.c reads them."""
    sd = model.state_dict()
    c = model.config
    dim = c.hidden_size

    def put(f, name):
        f.write(sd[name].detach().cpu().numpy().astype("<f4").tobytes())

    with open(path, "wb") as f:
        f.write(struct.pack("<7i", MAGIC_CLS, dim, c.num_hidden_layers, c.num_attention_heads,
                            c.intermediate_size, c.vocab_size, c.max_position_embeddings))
        put(f, "embeddings.word_embeddings.weight")
        put(f, "embeddings.position_embeddings.weight")
        # all our tokens have type 0, so only the first row is needed
        f.write(sd["embeddings.token_type_embeddings.weight"][0].detach().cpu().numpy().astype("<f4").tobytes())
        put(f, "embeddings.LayerNorm.weight")
        put(f, "embeddings.LayerNorm.bias")
        for i in range(c.num_hidden_layers):
            p = f"encoder.layer.{i}."
            for part in ["attention.self.query", "attention.self.key", "attention.self.value",
                         "attention.output.dense", "attention.output.LayerNorm",
                         "intermediate.dense", "output.dense", "output.LayerNorm"]:
                put(f, p + part + ".weight")
                put(f, p + part + ".bias")


@torch.no_grad()
def embed(model, tokenizer, texts, max_len=256):
    """Take the [CLS] token and scale it to length 1, like this model expects."""
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        vec = model(**enc).last_hidden_state[0, 0]
        out.append((vec / vec.norm()).cpu().numpy())
    return np.stack(out).astype("<f4")


def main():
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    export_model(model, "bge.bin")
    vocab = tokenizer.get_vocab()
    with open("vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(vocab, key=vocab.get)) + "\n")

    news = load_dataset("fancyzhx/ag_news", split="test")
    with open("texts.txt", "w", encoding="utf-8") as f:
        for row in news.select(range(N_TEXTS)):
            f.write(" ".join(row["text"].split()).replace("\\", " ") + "\n")

    with open("check.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(CHECK_TEXTS) + "\n")
    embed(model, tokenizer, CHECK_TEXTS).tofile("check.bin")

    with tarfile.open("bge_data.tar", "w") as tar:
        for name in ["bge.bin", "vocab.txt", "texts.txt", "check.txt", "check.bin"]:
            tar.add(name)
    print("Done: download bge_data.tar")


if __name__ == "__main__":
    main()
