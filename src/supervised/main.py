import os
from supervised_clustering import run_matching_for_all_models

def load_events_from_data_dir():
    """Reprend la logique de ton script original : lit la dernière colonne
    de chaque CSV dans le dossier ../data."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    event_list = []

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            value = line.split(",")[-1].strip().strip('"')
            if value:
                event_list.append(value)
    return event_list


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Charge tes événements bruts
    event_list_raw = load_events_from_data_dir()
    print(f"{len(event_list_raw)} événements chargés")

    if not event_list_raw:
        print("Aucun événement chargé, arrêt.")
        raise SystemExit(0)

    # 2) Matching zero-shot contre tes labels connus, pour chaque modèle
    all_results = run_matching_for_all_models(event_list_raw)

    # 3) -> Ouvre les CSV générés dans output/, remplis "label_corrige"
    #    pour les lignes "uncertain" (et corrige celles que tu juges fausses).

    # 4) Une fois corrigé, décommente pour construire le dataset d'entraînement
    #    et lancer le fine-tuning SetFit sur le modèle qui t'a donné le
    #    meilleur score moyen / le plus de "confident" à l'étape 2.
    #
    # best_model_name = "dangvantuan/sentence-camembert-large"
    # df_corrected = pd.read_csv(
    #     os.path.join(OUTPUT_DIR, f"matching_{best_model_name.replace('/', '_')}.csv")
    # )
    # df_train = build_training_dataset(df_corrected)
    # print(df_train["label"].value_counts())
    # finetune_setfit(df_train, best_model_name)