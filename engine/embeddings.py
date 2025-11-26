# embeddings.py
# Handles all embedding operations using sentence-transformers (MiniLM model)

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingEngine:
    """
    Wrapper class for generating sentence embeddings using a local MiniLM model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Loads the embedding model.
        """
        print(f"[EmbeddingEngine] Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("[EmbeddingEngine] Model loaded successfully.")

    def encode(self, sentences):
        """
        Converts a list of texts into embeddings.
        :param sentences: List of strings.
        :return: numpy array of shape (n_samples, 384)
        """
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of sentences.")

        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        return np.array(embeddings)

    def encode_single(self, sentence: str):
        """
        Encodes a single sentence.
        """
        return self.encode([sentence])[0]
