from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pickle
import pandas as pd
import torch
from typing import Tuple, List, Optional

from abc import ABC, abstractmethod

from src.data import BaseData, VectorizedData
from src.summary import summarize


class Cluster:
    def __init__(
        self, cluster_id: int, center, chunks: List[str], embeddings: torch.Tensor
    ) -> None:
        self.cluster_id = cluster_id
        self.center = center
        self.chunks = chunks
        self.embeddings = embeddings

        self.summary = None

    def get_summary(self):
        if self.summary is None:
            self.summary = summarize(" ".join(self.chunks[:3]))
        return self.summary


class Clusterizator(ABC):
    def __init__(self, vectorized_data: VectorizedData) -> None:
        self.vectorized_data = vectorized_data
        self.chunks = self.vectorized_data.get_chunks()
        self.embeddings = self.vectorized_data.get_embeddings()
        self.clusters = {}

    def clusterize(
        self,
        num_clusters: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        if embeddings is None:
            embeddings = self.embeddings

        self._clusterize(embeddings, num_clusters)
        self._set_clusters()

    def _set_clusters(self):
        for cluster_id, center in enumerate(self.get_clusters_centers()):
            cluster_chunks, cluster_vectors = self.get_vectors_by_cluster_id(cluster_id)
            cluster = Cluster(cluster_id, center, cluster_chunks, cluster_vectors)

            self.clusters[cluster_id] = cluster

    def get_cluster_by_id(self, cluster_id) -> Cluster:
        return self.clusters[cluster_id]

    def get_cluster_summaries(self):
        summaries = {}
        for cluster_id, cluster in self.clusters.items():
            summaries[cluster_id] = cluster.get_summary()
        return summaries

    @abstractmethod
    def _clusterize(self, embeddings: torch.Tensor):
        ...

    @abstractmethod
    def get_clusters_centers(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_vectors_by_cluster_id(self, cluster_id, top_k=None):
        ...

    def get_samples(self, top_k=3):
        samples = []
        for cluster_id, center in enumerate(self.get_clusters_centers()):
            sample_cluster = self.get_vectors_by_cluster_id(cluster_id, top_k=top_k)
            samples.append(sample_cluster)
        return samples


class KMeansClusterizator(Clusterizator):
    def determine_k(self, embeddings):
        k_min = 10
        clusters = [x for x in range(2, k_min * 11)]
        metrics = []
        for i in clusters:
            metrics.append((KMeans(n_clusters=i, n_init=10).fit(embeddings)).inertia_)

        k = self.elbow(k_min, clusters, metrics)
        return k

    def elbow(self, k_min, clusters, metrics):
        score = []

        for i in range(k_min, clusters[-3]):
            y1 = np.array(metrics)[: i + 1]
            y2 = np.array(metrics)[i:]

            df1 = pd.DataFrame({"x": clusters[: i + 1], "y": y1})
            df2 = pd.DataFrame({"x": clusters[i:], "y": y2})

            reg1 = LinearRegression().fit(np.asarray(df1.x).reshape(-1, 1), df1.y)
            reg2 = LinearRegression().fit(np.asarray(df2.x).reshape(-1, 1), df2.y)

            y1_pred = reg1.predict(np.asarray(df1.x).reshape(-1, 1))
            y2_pred = reg2.predict(np.asarray(df2.x).reshape(-1, 1))

            score.append(
                mean_squared_error(y1, y1_pred) + mean_squared_error(y2, y2_pred)
            )

        return np.argmin(score) + k_min

    def _clusterize(self, num_clusters: Optional[int], embeddings: np.ndarray):
        if num_clusters is None:
            num_clusters = self.determine_k(embeddings)

        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
        self.kmeans_labels = self.kmeans.labels_

    def get_clusters_centers(self) -> np.ndarray:
        return self.kmeans.cluster_centers_

    def get_vectors_by_cluster_id(
        self, cluster_id: int, top_k: int = None
    ) -> Tuple[List[str], List[np.ndarray]]:
        data = pd.DataFrame()
        data["text"] = self.chunks
        data["label"] = self.kmeans_labels
        data["embedding"] = list(self.embeddings)

        kmeans_centers = self.get_clusters_centers()

        cluster = data[data["label"].eq(cluster_id)]
        embeddings = list(cluster["embedding"])
        texts = list(cluster["text"])
        distances = [
            cosine_similarity(kmeans_centers[0].reshape(1, -1), e.reshape(1, -1))[0][0]
            for e in embeddings
        ]
        scores = list(zip(texts, distances))

        if top_k is None:
            top_k = len(scores)

        top_k = min(len(scores), top_k)

        top = sorted(scores, key=lambda x: x[1])[:top_k]
        top_texts = list(zip(*top))[0]
        return top_texts, embeddings

    def save(self, pth: str = "data/clusters.pkl"):
        with open(pth, "wb") as model_file:
            pickle.dump(self.kmeans, model_file)

    def load(self, pth: str = "data/clusters.pkl"):
        with open(pth, "rb") as model_file:
            self.kmeans = pickle.load(model_file)
            self.kmeans_labels = self.kmeans.labels_

        self._set_clusters()


class DBSCANClusterizator(Clusterizator):
    ...
