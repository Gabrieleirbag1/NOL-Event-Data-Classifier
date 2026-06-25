import os
import pandas as pd
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..","output", "supervised")

def build_training_dataset(df, use_corrected=True):
    """
    À partir d'un DataFrame exporté (matching_<model>.csv que tu as ouvert
    et corrigé dans Excel via la colonne 'label_corrige'), construit le
    dataset final pour l'entraînement.

    Règle : si 'label_corrige' est rempli -> on l'utilise (vérité terrain).
            sinon, si 'status' == 'confident' -> on garde la prédiction.
            sinon -> on ignore la ligne (trop incertaine, non corrigée).
    """
    df = df.copy()

    def pick_label(row):
        if use_corrected and isinstance(row.get("label_corrige"), str) and row["label_corrige"].strip():
            return row["label_corrige"].strip()
        if row["status"] == "confident":
            return row["predicted_label"]
        return None

    df["final_label"] = df.apply(pick_label, axis=1)
    df_train = df[df["final_label"].notna()][["event", "final_label"]].rename(
        columns={"event": "text", "final_label": "label"}
    )
    return df_train.reset_index(drop=True)


def finetune_setfit(df_train, base_model_name, output_name=None, test_size=0.2):
    """
    Fine-tune un sentence-transformer avec SetFit sur le dataset construit.
    Nécessite : pip install setfit datasets
    """
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset
    from sklearn.metrics import classification_report

    if output_name is None:
        output_name = f"setfit_{base_model_name.replace('/', '_')}"

    dataset = Dataset.from_pandas(df_train)
    split = dataset.train_test_split(test_size=test_size, seed=42)

    model = SetFitModel.from_pretrained(base_model_name)
    args = TrainingArguments(
        num_epochs=3,
        batch_size=16,
        num_iterations=20,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        metric="accuracy",
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Accuracy sur le test set : {metrics['accuracy']:.3f}")

    preds = model.predict(split["test"]["text"])
    print(classification_report(split["test"]["label"], preds, zero_division=0))

    save_path = os.path.join(OUTPUT_DIR, output_name)
    model.save_pretrained(save_path)
    print(f"Modèle sauvegardé : {save_path}")

    return model, metrics

if __name__ == "__main__":
    best_model_name = "paraphrase-multilingual-mpnet-base-v2"
    df_corrected = pd.read_csv(
        os.path.join(OUTPUT_DIR, f"matching_{best_model_name.replace('/', '_')}.csv")
    )
    df_train = build_training_dataset(df_corrected)
    print(df_train["label"].value_counts())
    finetune_setfit(df_train, best_model_name)