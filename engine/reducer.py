# reducer.py
# Handles dimensionality reduction using UMAP

import umap
import numpy as np


class UMAPReducer:
    """
    Reduces high-dimensional embeddings (e.g., 384-d) into 2D or 3D space using UMAP.
    """

    def __init__(self,
                 n_neighbors: int = 12,
                 min_dist: float = 0.1,
                 n_components: int = 2,
                 random_state: int = 42):
        """
        Initializes the UMAP reducer.
        """
        print(f"[UMAPReducer] Initializing UMAP with {n_neighbors=}, {min_dist=}, "
              f"{n_components=}, {random_state=}")

        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric="cosine",
            random_state=random_state
        )

    def fit_transform(self, embeddings: np.ndarray):
        """
        Reduces embeddings to low-dimensional form.
        :param embeddings: numpy array (n_samples, embedding_dim)
        :return: numpy array (n_samples, n_components)
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy ndarray.")

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D. Got shape {embeddings.shape}")

        print(f"[UMAPReducer] Reducing {embeddings.shape[0]} samples from "
              f"{embeddings.shape[1]}D → {self.reducer.n_components}D")

        return self.reducer.fit_transform(embeddings)
