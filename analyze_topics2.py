import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from umap.umap_ import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

# 1 load data
df = pd.read_csv("survey_responses_24_09.csv")
texts = df["ruclearnex"].dropna().astype(str).tolist()
texts = [text.strip() for text in texts if text.strip()]  # basic cleanup

print(f"Number of responses: {len(texts)}")

# 2 Shared components
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer()
representation_model = KeyBERTInspired()

E = embedding_model.encode(texts, normalize_embeddings=True)
E = np.asarray(E)

# 3 parameter grid
n_neighbors_list = [20, 22, 25, 27, 30]
n_components_list = [3,4, 5]            
min_cluster_size_list = [ 21, 22, 23, 24, 25, 26, 27]

results = []

def evaluate_clustering(embeddings, labels):
    """Compute cluster quality metrics, skipping outliers (-1)."""
    labels = np.asarray(labels)
    mask = labels != -1
    E_in = embeddings[mask]
    y_in = labels[mask]

    # if # of clusters is too small, it really isn't useful, so here I require at least 4
    unique_labels, counts = np.unique(y_in, return_counts=True)
    if len(unique_labels) < 4 or np.any(counts < 2):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "separability": np.nan,
            "combined": np.nan
        }

    try:
        sil = silhouette_score(E_in, y_in, metric="cosine")
        db  = davies_bouldin_score(E_in, y_in)
        ch  = calinski_harabasz_score(E_in, y_in)

        # centroid separability
        centroids = {t: E_in[y_in == t].mean(axis=0) for t in np.unique(y_in)}
        intra = np.mean([
            np.linalg.norm(E_in[y_in == t] - centroids[t], axis=1).mean()
            for t in centroids
        ])
        inter = np.mean([
            np.linalg.norm(c1 - c2)
            for i, c1 in enumerate(centroids.values())
            for j, c2 in enumerate(centroids.values()) if j > i
        ])
        separability = inter / intra if intra > 0 else np.nan
        comb = (
        0.65 * sil +
        0.2 * separability +
        0.05  * ch -
        0.1  * db
        )
        

    except Exception as e:
        print(f"Metric computation failed: {e}")
        sil = db = ch = separability = np.nan

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "separability": separability,
        "combined" : comb
    }

# 4 search loop
run_id = 0
for n_neighbors in n_neighbors_list:
    for n_components in n_components_list:
        for min_cluster_size in min_cluster_size_list:
            run_id += 1
            print(
                f"\n=== Run {run_id} === "
                f"UMAP(n_neighbors={n_neighbors}, n_components={n_components}), "
                f"HDBSCAN(min_cluster_size={min_cluster_size})"
            )

            # UMAP and HDBSCAN with current params
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.1,
                metric="cosine"
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True
            )

            # Build model for this configuration
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                representation_model=representation_model
            )

            topics, probs = topic_model.fit_transform(texts)

            topic_labels = np.asarray(topics)
            n_topics = len(set(topic_labels)) - (1 if -1 in topic_labels else 0)
            print(f"Number of non-outlier topics: {n_topics}")

            # evaluate clustering quality 
            metrics = evaluate_clustering(E, topic_labels)

            print(
                f"Silhouette: {metrics['silhouette']:.4f}, "
                f"DB: {metrics['davies_bouldin']:.4f}, "
                f"CH: {metrics['calinski_harabasz']:.2f}, "
                f"Separability: {metrics['separability']:.4f}"
                f"Combined: {metrics['combined']:.4f}"
            )

            # store results
            results.append({
                "run_id": run_id,
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_cluster_size": min_cluster_size,
                "n_topics": n_topics,
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "separability": metrics["separability"],
                "combined": metrics['combined'],
                "model": topic_model,   # keep the fitted model
            })

# 5 pick best configuration 
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(
    by=["combined"],
    ascending=[False]
)
print("\n=== All runs sorted by performance ===")
print(
    results_df_sorted[
        ["run_id", "n_neighbors", "n_components", "min_cluster_size",
         "n_topics", "silhouette", "davies_bouldin", "calinski_harabasz",
         "separability", "combined"]
    ]
)

best_result = results_df_sorted.iloc[0]
best_model = best_result["model"]

print("\n=== Best configuration ===")
print(best_result[
    ["run_id", "n_neighbors", "n_components", "min_cluster_size",
     "n_topics", "silhouette", "davies_bouldin", "calinski_harabasz",
     "separability", "combined"]
])

# 6 visualize and save
best_model.visualize_topics().show()

best_model.save(
    "best_topic_model",
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
