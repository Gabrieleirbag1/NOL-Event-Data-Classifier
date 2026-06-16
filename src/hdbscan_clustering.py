import os
import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import csv

MODELS = [
    ("paraphrase-multilingual-mpnet-base-v2", "sentence_transformer"),
    ("dangvantuan/sentence-camembert-large",   "sentence_transformer"),
]

def save_clusters_to_file(model_name, clusters):
    base = os.path.join(os.path.dirname(__file__), "..", "output")
    safe_name = model_name.replace('/', '_')

    # ── JSON ─────────────────────────────────────────────
    json_data = {
        "model": model_name,
        "clusters": [
            {
                "cluster": cluster_name,
                "count": len(events),
                "events": events
            }
            for cluster_name, events in sorted(clusters.items())
        ]
    }
    with open(os.path.join(base, f"clusters_{safe_name}.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # ── CSV ──────────────────────────────────────────────
    with open(os.path.join(base, f"clusters_{safe_name}.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "cluster_name", "event"])
        writer.writeheader()
        for cluster_name, events in sorted(clusters.items()):
            cluster_id = cluster_name.replace("Cluster ", "") if "Cluster" in cluster_name else "-1"
            for event in events:
                writer.writerow({
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "event": event
                })

def display_clusters(event_list, labels, embeddings, model_name):
    # Réduction en 2D avec PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])

    palette = sns.color_palette("tab20", n_clusters)
    color_map = {}
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            color_map[label] = (0.6, 0.6, 0.6)  # gris pour outliers
        else:
            color_map[label] = palette[color_idx % len(palette)]
            color_idx += 1

    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.set_theme(style="whitegrid")

    # Points
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=80,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5
    )

    # Labels texte sur chaque point
    # for i, (x, y) in enumerate(coords):
    #     ax.text(
    #         x, y + 0.012,
    #         event_list[i],
    #         fontsize=7,
    #         ha="center",
    #         va="bottom",
    #         alpha=0.75
    #     )

    # Légende
    legend_patches = []
    for label in unique_labels:
        name = f"Cluster {label}" if label != -1 else "Outliers"
        legend_patches.append(mpatches.Patch(color=color_map[label], label=name))
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    ax.set_title(f"Clusters — {model_name}", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("PCA dimension 1", fontsize=9)
    ax.set_ylabel("PCA dimension 2", fontsize=9)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..", "output", f"clusters_{model_name.replace('/', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Graphique sauvegardé : {out_path}")
    plt.show()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

def cluster_events(event_list, model_name, model_type):
    print(f"\n{'='*50}")
    print(f"Modèle : {model_name}")
    print(f"{'='*50}")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(event_list, show_progress_bar=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)

    clusters = defaultdict(list)
    for event, label in zip(event_list, labels):
        cluster_name = f"Cluster {label}" if label != -1 else "Outlier"
        clusters[cluster_name].append(event)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print(f"Résultats : {n_clusters} clusters | {n_outliers} outliers\n")

    save_clusters_to_file(model_name, clusters)
    display_clusters(event_list, labels, embeddings, model_name)

    return labels, clusters

def run_clustering(event_list):
    for model_name, model_type in MODELS:
        labels, clusters = cluster_events(event_list, model_name, model_type)