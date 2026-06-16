import os
import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict

MODELS = [
    ("paraphrase-multilingual-mpnet-base-v2", "sentence_transformer"),
    ("dangvantuan/sentence-camembert-large",   "sentence_transformer"),
    ("almanach/camembert-bio-gliner-v0.1",     "huggingface"),
]

def save_clusters_to_file(model_name, clusters):
    filename = os.path.join(os.path.dirname(__file__), "..", "output", f"clusters_{model_name.replace('/', '_')}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for cluster_name, events in sorted(clusters.items()):
            f.write(f"{cluster_name} ({len(events)} éléments)\n")
            for event in events:
                f.write(f"   - {event}\n")
            f.write("\n")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

def encode_with_huggingface(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

def cluster_events(event_list, model_name, model_type):
    print(f"\n{'='*50}")
    print(f"Modèle : {model_name}")
    print(f"{'='*50}")

    if model_type == "sentence_transformer":
        model = SentenceTransformer(model_name)
        embeddings = model.encode(event_list, show_progress_bar=True)
    elif model_type == "huggingface":
        embeddings = encode_with_huggingface(event_list, model_name)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)

    clusters = defaultdict(list)
    for event, label in zip(event_list, labels):
        cluster_name = f"Cluster {label}" if label != -1 else "⚠️ Non classifié (outlier)"
        clusters[cluster_name].append(event)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print(f"Résultats : {n_clusters} clusters | {n_outliers} outliers\n")

    # for cluster_name, events in sorted(clusters.items()):
    #     print(f"📂 {cluster_name} ({len(events)} éléments)")
    #     for event in events:
    #         print(f"   - {event}")
    #     print()
    save_clusters_to_file(model_name, clusters)

    return labels, clusters

def run_clustering(event_list):
    for model_name, model_type in MODELS:
        cluster_events(event_list, model_name, model_type)