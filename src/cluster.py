from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pickle
import pandas as pd


class Clusterizator:
    def __init__(self, chunks, embeddings) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        
    def clusterize(self, embeddings):
        ...
        
    def get_clusters_centers(self):
        ...
        
    def get_vectors_by_cluster_id(self, cluster_id, top_k=None):
        ...
        
    def get_samples(self, top_k=3):
        samples = []
        for cluster_id in range(len(self.get_clusters_centers())):
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
            y1 = np.array(metrics)[:i + 1]
            y2 = np.array(metrics)[i:]
        
            df1 = pd.DataFrame({'x': clusters[:i + 1], 'y': y1})
            df2 = pd.DataFrame({'x': clusters[i:], 'y': y2})
        
            reg1 = LinearRegression().fit(np.asarray(df1.x).reshape(-1, 1), df1.y)
            reg2 = LinearRegression().fit(np.asarray(df2.x).reshape(-1, 1), df2.y)

            y1_pred = reg1.predict(np.asarray(df1.x).reshape(-1, 1))
            y2_pred = reg2.predict(np.asarray(df2.x).reshape(-1, 1))    
            
            score.append(mean_squared_error(y1, y1_pred) + mean_squared_error(y2, y2_pred))

        return np.argmin(score) + k_min

    def clusterize(self):
        k_opt = self.determine_k(self.embeddings)
        self.kmeans = KMeans(n_clusters = k_opt, random_state = 42).fit(self.embeddings)
        self.kmeans_labels = self.kmeans.labels_
        
    def get_clusters_centers(self):
        return self.kmeans.cluster_centers_

    def get_vectors_by_cluster_id(self, cluster_id, top_k=None):
        data = pd.DataFrame()
        data['text'] = self.chunks
        data['label'] = self.kmeans_labels
        data['embedding'] = list(self.embeddings)

        kmeans_centers = self.get_clusters_centers()

        cluster = data[data['label'].eq(cluster_id)]
        embeddings = list(cluster['embedding'])
        texts = list(cluster['text'])
        distances = [cosine_similarity(kmeans_centers[0].reshape(1, -1), e.reshape(1, -1))[0][0] for e in embeddings]
        scores = list(zip(texts, distances))
        
        if top_k is None:
            top_k = len(scores)
            
        top_k = min(len(scores), top_k)
        
        top = sorted(scores, key=lambda x: x[1])[:top_k]
        top_texts = list(zip(*top))[0]
        return top_texts
    
    def save(self, pth):
        with open(pth, 'wb') as model_file:
            pickle.dump(self.kmeans, model_file)

    def load(self, pth):
        with open(pth, 'rb') as model_file:
            self.kmeans = pickle.load(model_file)
            self.kmeans_labels = self.kmeans.labels_