import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output", "supervised")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "paraphrase-multilingual-mpnet-base-v2",
    "dangvantuan/sentence-camembert-large",
]

CONFIDENCE_THRESHOLD = 0.55

RAW_LABELS = [
    "Midazolam", "Xylo", "Propofol", "Rocu", "Ketamine",
    "Haldol 0.5, decadron 4",
    "Epidurale : PE (perfusion epidurale), BE (bolus epidurale)",
    "PR (perfusion rémi) (ex :br20 + pr up 0.04) PR up ou down",
    "BR (bolus rémi)",
    "PP (perfusion phenil) (ex : bp 80 + pp up 0.5) PP up ou down",
    "BP (bolus phenil)",
    "Levophed : PL ou BL",
    "3 min baseline", "5 min no touch", "Stim", "PREOX",
    "Laryngo + intubation", "Manual ventilation (MV)", "Guedel",
    "Start sevo", "positionnement", "Art line", "IV line",
    "Sonde urinaire", "Skin disinfection", "Draping (champs)",
    "Infiltration", "Incision", "Insufflation", "Exsufflation",
    "TNG (tube nasogastrique)", "Debut fermeture plan profond (DFPP)",
    "Dilaudid", "Zofran", "Fin fermeture peau (FFP)", "Agrafes",
    "Renverse (ex : néo 3.0 + glyco 0.4 ou sugammadex 300)",
    "TOF 4/4 100%", "Stop remi", "Stop phenil", "Stop sevo",
    "Aide inspi", "Aspiration buccale", "MAN/SPON", "Extubation",
]

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
    # /, %, +, :, etc. are often just noise
    lowered = re.sub(r"[/%+:]", " ", lowered)
    lowered = re.sub(r"\s*,\s*", " ", lowered)

    # Param words
    words = [w for w in lowered.split() if w not in PARAM_WORDS]
    lowered = " ".join(words)

    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered if lowered else text.lower()  # fallback si tout est supprimé

def clean_label(label):
    short = re.split(r"[:\(]", label)[0].strip()
    short = short if short else label
    return clean_text(short)

def match_events_to_labels(event_list_raw, labels_raw, model_name, top_k=3):
    model = SentenceTransformer(model_name)

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

    return results

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

def run_matching_for_all_models(event_list_raw, labels_raw=RAW_LABELS, models=MODELS):
    all_results = {}
    for model_name in models:
        print(f"\n{'='*60}\nModel : {model_name}\n{'='*60}")
        results = match_events_to_labels(event_list_raw, labels_raw, model_name)
        df = export_results(results, model_name)
        all_results[model_name] = df

    # Comparative summary : average score + number of confident predictions
    print(f"\n{'='*60}\nComparative Summary\n{'='*60}")
    for model_name, df in all_results.items():
        mean_score = df["score"].mean()
        n_confident = (df["status"] == "confident").sum()
        print(f"{model_name:45s} | avg score={mean_score:.3f} | confidents={n_confident}/{len(df)}")

    return all_results