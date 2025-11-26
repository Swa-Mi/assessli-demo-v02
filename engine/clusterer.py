# clusterer.py
# Adaptive KMeans clusterer optimized for small datasets (10-200 samples).

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ClusterEngine:
    """
    Adaptive KMeans clustering engine that chooses k using a hybrid method:
      - elbow/knee method on inertia
      - density-based estimate using DBSCAN over multiple eps values
      - silhouette-based best-k
    Final k is the rounded average of the three signals (clamped to [2, max_k]).
    Designed for small datasets (10 - 200 samples).
    """

    def __init__(self, min_k: int = 2, max_k: int = 12, random_state: int = 42):
        self.min_k = max(2, int(min_k))
        self.max_k = max(int(self.min_k), int(max_k))
        self.random_state = random_state
        self.model = None
        print(
            f"[ClusterEngine] Adaptive KMeans initialized (min_k={self.min_k}, max_k={self.max_k})")

    def _safe_range(self, n_samples):
        """Return sensible max_k based on number of samples."""
        effective_max = min(self.max_k, max(self.min_k, n_samples - 1))
        return effective_max

    # -------------------------
    # Signal 1: Elbow / Knee
    # -------------------------
    def _compute_elbow_k(self, embeddings: np.ndarray, max_k: int):
        """
        Compute inertia for k=2..max_k and detect elbow using max distance from line method.
        Returns elbow_k (int). Falls back to min_k if something fails.
        """
        try:
            inertias = []
            K = list(range(self.min_k, max_k + 1))
            for k in K:
                kmeans = KMeans(
                    n_clusters=k, random_state=self.random_state, n_init="auto")
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)

            # Normalize points for knee detection
            xs = np.array(K, dtype=float)
            ys = np.array(inertias, dtype=float)
            # line from first to last
            p1 = np.array([xs[0], ys[0]])
            p2 = np.array([xs[-1], ys[-1]])
            # distances from line

            def point_line_distance(px, p1, p2):
                # cross product area / base length
                return np.abs(np.cross(p2 - p1, px - p1) / np.linalg.norm(p2 - p1) + 1e-12)

            dists = np.array([point_line_distance(
                np.array([x, y]), p1, p2) for x, y in zip(xs, ys)])
            elbow_idx = int(np.argmax(dists))
            elbow_k = int(xs[elbow_idx])
            # sanity clamp
            elbow_k = max(self.min_k, min(elbow_k, max_k))
            return elbow_k
        except Exception:
            return self.min_k

    # -------------------------
    # Signal 2: Density-based guess using DBSCAN
    # -------------------------
    def _compute_density_k(self, embeddings: np.ndarray, max_k: int):
        """
        Run DBSCAN across several eps multipliers of median nearest-neighbor distance.
        For each eps, count clusters (excluding noise). Return a suggested k (median of valid counts).
        If DBSCAN finds <=1 clusters for all eps, fallback to min_k.
        """
        try:
            n_samples = len(embeddings)
            if n_samples < 5:
                return self.min_k

            # compute median distance to 3rd nearest neighbor
            nn = NearestNeighbors(n_neighbors=min(
                5, n_samples - 1)).fit(embeddings)
            dists, _ = nn.kneighbors(embeddings)
            # take the 3rd neighbor if available, else last
            idx = min(3, dists.shape[1] - 1)
            third_nn = dists[:, idx]
            base_eps = float(np.median(third_nn) + 1e-12)

            eps_factors = [0.6, 0.8, 1.0, 1.2, 1.5]
            cluster_counts = []
            for f in eps_factors:
                eps = base_eps * f
                db = DBSCAN(eps=eps, min_samples=2,
                            metric="euclidean").fit(embeddings)
                labels = db.labels_
                # count clusters excluding noise (-1)
                unique = set(labels)
                if -1 in unique:
                    unique.remove(-1)
                cnt = len(unique)
                if cnt >= 1:
                    cluster_counts.append(cnt)

            if not cluster_counts:
                return self.min_k

            # clip counts to [min_k, max_k]
            clipped = [max(self.min_k, min(c, max_k)) for c in cluster_counts]
            # return median suggestion
            density_k = int(np.median(clipped))
            return density_k
        except Exception:
            return self.min_k

    # -------------------------
    # Signal 3: Silhouette best k
    # -------------------------
    def _compute_silhouette_k(self, embeddings: np.ndarray, max_k: int):
        """
        Compute silhouette scores for k in [min_k..max_k] and return the k with highest silhouette.
        Fall back to min_k if silhouette cannot be computed (e.g., too few samples).
        """
        try:
            n_samples = len(embeddings)
            if n_samples < 4:
                return self.min_k

            best_k = self.min_k
            best_score = -1.0
            for k in range(self.min_k, max_k + 1):
                if k >= n_samples:
                    break
                kmeans = KMeans(
                    n_clusters=k, random_state=self.random_state, n_init="auto")
                labels = kmeans.fit_predict(embeddings)
                # silhouette requires at least 2 clusters and fewer clusters than samples
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            return best_k
        except Exception:
            return self.min_k

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, embeddings: np.ndarray):
        """
        Fit the clustering engine and return integer labels for each embedding.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array.")

        n_samples = len(embeddings)
        if n_samples < 2:
            # trivial case: single cluster label 0
            return np.array([0] * n_samples)

        max_k = self._safe_range(n_samples)

        # Compute three signals
        elbow_k = self._compute_elbow_k(embeddings, max_k)
        density_k = self._compute_density_k(embeddings, max_k)
        silhouette_k = self._compute_silhouette_k(embeddings, max_k)

        # Combine signals (weighted average) - give silhouette slightly more weight
        suggested = int(
            round((0.25 * elbow_k) + (0.25 * density_k) + (0.5 * silhouette_k)))
        # clamp to valid range and to n_samples-1
        final_k = max(self.min_k, min(suggested, max_k, n_samples - 1))

        # Final KMeans clustering
        self.model = KMeans(n_clusters=final_k,
                            random_state=self.random_state, n_init="auto")
        labels = self.model.fit_predict(embeddings)

        print(
            f"[ClusterEngine] Signals -> elbow_k={elbow_k}, density_k={density_k}, silhouette_k={silhouette_k}")
        print(
            f"[ClusterEngine] Final chosen k = {final_k} (n_samples={n_samples})")
        return labels

    def get_cluster_dict(self, texts, labels):
        """
        Groups original texts under their assigned cluster.
        """
        clusters = defaultdict(list)
        for text, label in zip(texts, labels):
            clusters[int(label)].append(text)
        return clusters

    def soft_membership_scores(self):
        """
        KMeans is hard-clustering; no soft membership available here.
        Return None for compatibility with previous interface.
        """
        print(
            "[ClusterEngine] Note: KMeans does not provide soft membership (returning None).")
        return None
