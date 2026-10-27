"""Text embeddings for policy search: BAAI/bge-small-en-v1.5 on ONNX Runtime.

fastembed runs the model with onnxruntime, so there's no PyTorch in the image
(the same reasoning as the vision service). 384-dim vectors, 512-token input,
MIT licence. Ingest and search must use the same model, which is why both
import it from here.
"""

import os

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
# BGE is trained with this prefix on short queries (not on passages); it helps
# a question like "is hail covered?" match clause text. fastembed doesn't add it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, cache_dir=None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            MODEL, cache_dir=cache_dir or os.getenv("EMBED_CACHE_DIR")
        )

    def passages(self, texts):
        return list(self.model.passage_embed(texts))

    def query(self, text):
        return next(iter(self.model.query_embed(QUERY_INSTRUCTION + text)))
