import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "supervised")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    # "dangvantuan/sentence-camembert-large",
    "paraphrase-multilingual-mpnet-base-v2",
    "setfit_paraphrase-multilingual-mpnet-base-v2-1-1"
]

CONFIDENCE_THRESHOLD = 0.55

with open(os.path.join(os.path.dirname(__file__), "labels.json"), "r", encoding="utf-8") as f:
    RAW_LABELS = json.load(f)

PARAM_WORDS = ("up", "down", "start", "stop")

def clean_text(text):
    text = text.strip()
    lowered = text.lower()
    # 5mg, 200ml, 0.04 ...
    lowered = re.sub(r"\d+[\.,]?\d*\s*(mg|ml|g|kg|cc|ui|µg|mcg)\b", "", lowered, flags=re.IGNORECASE)
    # 4/4, 100%, br20
    lowered = re.sub(r"\d+\s*/\s*\d+", "", lowered)
    lowered = re.sub(r"\d+\s*%", "", lowered)
    # Other numbers
    lowered = re.sub(r"\d+[\.,]?\d*", "", lowered)
    # /, %, :, etc. are often just noise
    lowered = re.sub(r"[/%:]", " ", lowered)
    lowered = re.sub(r"\s*,\s*", " ", lowered)

    # Param words
    words = [w for w in lowered.split() if w not in PARAM_WORDS]
    lowered = " ".join(words)

    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered if lowered else text.lower()  # fallback if everything was removed


def clean_label(label):
    short = re.split(r"[:\(]", label)[0].strip()
    short = short if short else label
    return clean_text(short)


def match_events_to_labels(event_list_raw, labels_raw, model_name, model_path=None, top_k=3):
    model: SentenceTransformer = SentenceTransformer(model_path if model_path else model_name)

    events_clean = [clean_text(e) for e in event_list_raw]
    labels_clean = [clean_label(l) for l in labels_raw]

    label_embeddings = model.encode(labels_clean, show_progress_bar=False)
    event_embeddings = model.encode(events_clean, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    label_embeddings = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)
    event_embeddings = event_embeddings / np.linalg.norm(event_embeddings, axis=1, keepdims=True)

    similarity_matrix = event_embeddings @ label_embeddings.T  # (n_events, n_labels)

    results = []
    for i, event_raw in enumerate(event_list_raw):
        sims = similarity_matrix[i]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results.append({
            "event_raw": event_raw,
            "event_clean": events_clean[i],
            "best_label": labels_raw[top_indices[0]],
            "best_score": float(sims[top_indices[0]]),
            "top_k": [
                {"label": labels_raw[idx], "score": float(sims[idx])}
                for idx in top_indices
            ],
        })

    return results, event_embeddings


def export_results(results, model_name, threshold=CONFIDENCE_THRESHOLD):
    safe_name = model_name.replace("/", "_")

    rows = []
    for r in results:
        rows.append({
            "event": r["event_raw"],
            "predicted_label": r["best_label"],
            "score": round(r["best_score"], 4),
            "status": "confident" if r["best_score"] >= threshold else "uncertain",
            "alt_label_2": r["top_k"][1]["label"] if len(r["top_k"]) > 1 else "",
            "alt_score_2": round(r["top_k"][1]["score"], 4) if len(r["top_k"]) > 1 else "",
            "alt_label_3": r["top_k"][2]["label"] if len(r["top_k"]) > 2 else "",
            "alt_score_3": round(r["top_k"][2]["score"], 4) if len(r["top_k"]) > 2 else "",
            # Correct manually
            "label_corrige": "",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f"matching_{safe_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    json_path = os.path.join(OUTPUT_DIR, f"matching_{safe_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_confident = (df["status"] == "confident").sum()
    n_uncertain = (df["status"] == "uncertain").sum()
    print(f"  -> {csv_path}")
    print(f"     {n_confident} Confidents / {n_uncertain} Uncertains (threshold={threshold})")

    return df


def display_clusters_graph(df, embeddings, model_name, show=False, min_label_count=2, max_text_labels=60):
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"df has {len(df)} rows but embeddings has {embeddings.shape[0]} "
            "— they must correspond row by row."
        )

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()

    label_counts = df["predicted_label"].value_counts()
    rare_labels = set(label_counts[label_counts < min_label_count].index)

    plot_labels = df["predicted_label"].apply(
        lambda l: "Others (rare)" if l in rare_labels else l
    )
    unique_labels = sorted(plot_labels.unique())

    n_colors_needed = len([l for l in unique_labels if l != "Others (rare)"])
    palette = sns.color_palette("tab20", max(n_colors_needed, 1))
    color_map = {}
    idx = 0
    for label in unique_labels:
        if label == "Others (rare)":
            color_map[label] = (0.6, 0.6, 0.6)
        else:
            color_map[label] = palette[idx % len(palette)]
            idx += 1

    colors = [color_map[l] for l in plot_labels]

    edge_colors = ["red" if s == "uncertain" else "black" for s in df["status"]]
    edge_widths = [1.4 if s == "uncertain" else 0.5 for s in df["status"]]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, alpha=0.8, s=110,
        edgecolors=edge_colors, linewidths=edge_widths,
    )

    if len(df) <= max_text_labels:
        for (x, y), event_text in zip(coords, df["event"]):
            ax.annotate(
                str(event_text), (x, y),
                fontsize=7, alpha=0.75,
                xytext=(0, 6), textcoords="offset points", ha="center",
            )

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor=color_map[label], markersize=10)
        for label in unique_labels
    ]
    ax.legend(handles=handles, title="Predicted label", loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=8)

    ax.set_title(
        f"Matching events -> business labels — {model_name}\n"
        f"(PCA, explained variance = {explained:.1%}, "
        f"red edge = uncertain prediction)",
        fontsize=13,
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"clusters_{model_name.replace('/', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  -> graph saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)

    return out_path

def flatten_events(event_list_raw):
    flattened_events = []
    for event in event_list_raw:
        if "+" in event:
            type_match = re.search(r'^\s*(Type\s+\d+)', event, re.IGNORECASE)
            type_prefix = type_match.group(1).strip() if type_match else ""
            
            timestamp_match = re.search(r'(@\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\s*$', event)
            timestamp_suffix = timestamp_match.group(1).strip() if timestamp_match else ""
            
            middle_content = event
            if type_prefix:
                middle_content = re.sub(r'^\s*Type\s+\d+', '', middle_content, flags=re.IGNORECASE)
            if timestamp_suffix:
                middle_content = middle_content.rsplit(timestamp_suffix, 1)[0]
            
            sub_events = [e.strip() for e in middle_content.split("+") if e.strip()]
            
            for sub in sub_events:
                reconstructed = f"{type_prefix} {sub} {timestamp_suffix}".strip()
                reconstructed = re.sub(r'\s+', ' ', reconstructed)
                flattened_events.append(reconstructed)
        else:
            flattened_events.append(event)
    
    return flattened_events

def run_matching_for_all_models(event_list_raw, labels_raw=RAW_LABELS, models=MODELS, show_plots=False):
    # Separating the events that are concatenated with '+' into individual events
    event_list_raw = flatten_events(event_list_raw)

    all_results = {}
    for model_name in models:
        print(f"\n{'='*60}\nModel : {model_name}\n{'='*60}")

        model_path = os.path.join(MODEL_PATH, model_name.replace("/", "_"))
        if not os.path.exists(model_path):
            model_path = None

        results, event_embeddings = match_events_to_labels(event_list_raw, labels_raw, model_name, model_path)
        df = export_results(results, model_name)

        display_clusters_graph(df, event_embeddings, model_name, show=show_plots)

        all_results[model_name] = df

    # Comparative summary : average score + number of confident predictions
    print(f"\n{'='*60}\nComparative Summary\n{'='*60}")
    for model_name, df in all_results.items():
        mean_score = df["score"].mean()
        n_confident = (df["status"] == "confident").sum()
        print(f"{model_name:45s} | avg score={mean_score:.3f} | confidents={n_confident}/{len(df)}")

    return all_results