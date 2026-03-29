import pandas as pd 
import numpy as np

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap.umap_ import UMAP

from openai import OpenAI as OpenAIClient
from bertopic.representation import OpenAI as OpenAIRepresentation
import os

os.environ["OPENAI_API_KEY"] = ""
client = OpenAIClient()

# 1 load data
df = pd.read_csv("survey_responses_24_09.csv")
texts = df["ruclearnex"].dropna().astype(str).tolist()
texts = [text.strip() for text in texts if text.strip()]

print(f"Loaded {len(texts)} cleaned responses.")

# 2 set best params from search
BEST_PARAMS = {"n_neighbors": 20, "n_components": 4, "min_cluster_size": 27}
print("Using best parameters:", BEST_PARAMS)

# 3 build model

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=BEST_PARAMS["n_neighbors"],
    n_components=BEST_PARAMS["n_components"],
    min_dist=0.1,
    metric="cosine",
)

hdbscan_model = HDBSCAN(
    min_cluster_size=BEST_PARAMS["min_cluster_size"],
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer()

# here's the llm representation
prompt = """
You are labeling clusters of student feedback. 
Given the keywords and example responses below, generate a short topic label (3–6 words) that clearly represents the meaning.
Keywords:
[KEYWORDS]
Example Responses:
[DOCUMENTS]
Return ONLY the topic label, no explanations.
"""

representation_model = OpenAIRepresentation(
    client = client,
    model="gpt-4.1-mini",       
    prompt=prompt,
    max_tokens=20               
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,   
    verbose=True,
)

# 4 fit final model
topics, probs = topic_model.fit_transform(texts)

topic_labels = np.asarray(topics)
n_topics = len(set(topic_labels)) - (1 if -1 in topic_labels else 0)
print(f"Number of non-outlier topics: {n_topics}")

# 5 topic level information & export
topic_info = topic_model.get_topic_info()
print("\n=== Topic overview (head) ===")
print(topic_info.head())

topic_info.to_csv("best_model_topic_info_llm.csv", index=False)
print("Saved topic summary with LLM labels to best_model_topic_info_llm.csv")


docs_df = pd.DataFrame({
    "text": texts,
    "topic": topics,
    "probability": probs
})

docs_df.to_csv("best_model_documents_llm.csv", index=False)
print("Saved document-topic assignments to best_model_documents_llm.csv")

# 7 representative docs per topic
valid_mask = docs_df["topic"] != -1
docs_valid = docs_df[valid_mask].copy()

TOP_K = 10
top_docs_rows = []
for t in sorted(docs_valid["topic"].unique()):
    subset = docs_valid[docs_valid["topic"] == t]
    subset_sorted = subset.sort_values("probability", ascending=False).head(TOP_K)

    for rank, (_, row) in enumerate(subset_sorted.iterrows(), start=1):
        top_docs_rows.append({
            "topic": t,
            "rank_within_topic": rank,
            "probability": row["probability"],
            "text": row["text"],
        })

top_docs_df = pd.DataFrame(top_docs_rows)
top_docs_df.to_csv("best_model_top_docs_per_topic_llm.csv", index=False)
print(f"Saved top {TOP_K} representative docs per topic to best_model_top_docs_per_topic_llm.csv")

# 8 summary
print("\n=== Topic summary (ID, Size, LLM Label) ===")
for _, row in topic_info.iterrows():
    topic_id = row["Topic"]
    if topic_id == -1:
        continue  
    name = row.get("Name", "")
    count = row.get("Count", "")
    print(f"Topic {topic_id:>3} | Size: {count:>4} | Label: {name}")

# 9 save the trained model
topic_model.visualize_topics().show()
topic_model.save(
    "best_topic_model_final_llm",
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
print("Saved BERTopic model with LLM representation to best_topic_model_final_llm/")
