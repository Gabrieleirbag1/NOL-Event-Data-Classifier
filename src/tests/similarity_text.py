# Compare un texte avec et sans dose
from sentence_transformers import SentenceTransformer
import numpy as np

models = [
    "paraphrase-multilingual-mpnet-base-v2",
    "dangvantuan/sentence-camembert-large",
]

for model_name in models:
    print(f"\nModèle : {model_name}")
    model = SentenceTransformer(model_name)

    a = model.encode("ajout médicament 200mg")
    b = model.encode("ajout médicament 5mg")
    c = model.encode("incision aisselle 200mg")

    sim_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sim_ac = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))

    print(f"ajout 200mg vs ajout 5mg : {sim_ab:.3f}")
    print(f"ajout 200mg vs incision    : {sim_ac:.3f}")
    print(f"Différence : {sim_ab - sim_ac:.3f}")
    print(f"Ratio : {sim_ab / sim_ac:.3f}")
    print(" ")