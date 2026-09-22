import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedding:
    """
    Local sentence embedding using the fine-tuned
    multilingual-e5-small (384-dim).

    Used for both ingestion (batch) and query-time (single).
    """

    def __init__(
        self,
        model_name: str = "./e5-multilingual-small-law-deploy"
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            f"[Embedding] Loading {self.model_name} "
            f"on {self.device}..."
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        self.model.eval()

        print(
            "[Embedding] Ready. "
            f"Dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def get_embeddings(
        self,
        texts: list[str],
        prefix: str = "passage: ",
    ) -> torch.Tensor:
        """
        Embed a batch of strings.

        prefix:
            "query: "   for questions
            "passage: " for legal text

        Returns:
            Float32 tensor of shape (N, 384),
            L2-normalised, on CPU.
        """

        texts = [
            prefix + text
            for text in texts
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return torch.from_numpy(
            embeddings.astype(np.float32)
        )

    def _embed_one(
        self,
        text: str,
        prefix: str = "query: "
    ) -> list[float]:
        """
        Embed a single string and return
        it as a plain Python list.
        """

        return self.get_embeddings(
            [text],
            prefix=prefix
        )[0].tolist()