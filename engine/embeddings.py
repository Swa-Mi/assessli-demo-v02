# embeddings.py
# Lightweight embedding engine that works on Streamlit Cloud (NO PyTorch)

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingEngine:
    """
    Generates sentence embeddings using the lightweight MiniLM model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[EmbeddingEngine] Loading model (CPU-friendly): {model_name}")
        self.model = SentenceTransformer(model_name)
        print("[EmbeddingEngine] Model loaded successfully.")

    def encode(self, sentences):
        """
        Encodes a list of sentences.
        """
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of sentences.")

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True  # Ensures numpy output
        )
        return embeddings

    def encode_single(self, sentence: str):
        """
        Encodes a single sentence.
        """
        return self.encode([sentence])[0]
