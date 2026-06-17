import os
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import csv
import itertools
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from frequency_analysis import run_frequent_params_analysis

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
    #         event_list["content"][i],
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

def reduce_embeddings(embeddings, method="umap"):
    if method == "umap":
        reducer = UMAP(
            n_components=10,      # pas trop bas, 5-15 est bien
            n_neighbors=30,       # voisins considérés (+ grand = + global)
            min_dist=0.0,         # 0.0 = clusters plus compacts
            metric="cosine",      # cosine >> euclidean pour du texte
            random_state=42
        )
    reduced = reducer.fit_transform(embeddings)
    reduced = normalize(reduced, norm="l2")
    return reduced

def grid_search_hdbscan(embeddings):
    param_grid = {
        "min_cluster_size": [2, 3, 5, 10],
        "min_samples":      [1, 2, 5],
        "cluster_selection_method": ["eom", "leaf"],
    }

    best_score = -1
    best_params = None

    combos = list(itertools.product(*param_grid.values()))
    keys   = list(param_grid.keys())

    for combo in combos:
        params = dict(zip(keys, combo))
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)

        # Ignorer si trop peu de clusters ou trop d'outliers
        if n_clusters < 2 or n_outliers > len(embeddings) * 0.3:
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")

        print(f"  clusters={n_clusters} | outliers={n_outliers} | silhouette={score:.3f} | {params}")

        if score > best_score:
            best_score = score
            best_params = params

    print(f"\n Best params : {best_params} (silhouette={best_score:.3f})")
    return best_params

def assign_outliers_to_nearest_cluster(embeddings, labels):

    outlier_idx  = np.where(labels == -1)[0]
    cluster_idx  = np.where(labels != -1)[0]

    if len(outlier_idx) == 0:
        return labels

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(embeddings[cluster_idx])

    _, indices = nn.kneighbors(embeddings[outlier_idx])
    new_labels = labels.copy()
    for i, outlier in enumerate(outlier_idx):
        new_labels[outlier] = labels[cluster_idx[indices[i][0]]]

    return new_labels

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

def cluster_events(event_list, model_name, model_type):
    print(f"\n{'='*50}")
    print(f"Modèle : {model_name}")
    print(f"{'='*50}")

    #1 embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(event_list["normalized"], show_progress_bar=True)

    #2 reduction UMAP
    reduced = reduce_embeddings(embeddings, method="umap")

    # 3. Grid search
    best_params = grid_search_hdbscan(reduced)

    # 4. Clustering with best params
    clusterer = hdbscan.HDBSCAN(**best_params)
    labels = clusterer.fit_predict(embeddings)

    # 5. Réassigner les outliers
    labels = assign_outliers_to_nearest_cluster(reduced, labels)

    # 6. Show results
    clusters = defaultdict(list)
    for event, label in zip(event_list["content"], labels):
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
        run_frequent_params_analysis(clusters)