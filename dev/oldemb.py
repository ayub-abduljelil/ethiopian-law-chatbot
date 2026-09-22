import torch
from transformers import AutoTokenizer, AutoModel


class Embedding:
    """
    Local sentence embedding using MiniLM-L6-v2 (384-dim).
    Used for both ingestion (batch) and query-time (single).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedding] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("[Embedding] Ready.")

    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask:    torch.Tensor,
    ) -> torch.Tensor:
        mask      = attention_mask.unsqueeze(-1).float()
        summed    = (last_hidden_state * mask).sum(dim=1)
        counts    = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Embed a batch of strings.
        Returns a Float32 tensor of shape (N, 384), L2-normalised, on CPU.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single string and return it as a plain Python list."""
        return self.get_embeddings([text])[0].tolist()
