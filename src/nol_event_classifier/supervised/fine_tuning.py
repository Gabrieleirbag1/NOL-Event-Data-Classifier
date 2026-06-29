import os
import pandas as pd
from lite_logging.lite_logging import log
import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..","output", "supervised")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..","output", "models")

def build_training_dataset(df, use_corrected=True):
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

def finetune_setfit(df_train, base_model_name, output_name=None, test_size=0.2, num_epochs=1, batch_size=16, num_iterations=1):
    """
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
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_iterations=num_iterations,
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
    log(f"Accuracy sur le test set : {metrics['accuracy']:.3f}")

    preds = model.predict(split["test"]["text"])
    log(classification_report(split["test"]["label"], preds, zero_division=0))

    save_path = os.path.join(MODEL_DIR, output_name + f"_{num_epochs}ep-{batch_size}bs-{num_iterations}it")
    model.save_pretrained(save_path)
    log(f"Modèle sauvegardé : {save_path}")

    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a SetFit model on a training dataset")
    parser.add_argument("--base_model", "-m", type=str, default="paraphrase-multilingual-mpnet-base-v2", help="Base model name for SetFit")
    parser.add_argument("--output_name", "-o", type=str, default=None, help="Output name for the fine-tuned model")
    parser.add_argument("--test_size", "-t", type=float, default=0.2, help="Test size for train/test split")
    parser.add_argument("--num_epochs", "-e", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_iterations", "-i", type=int, default=1, help="Number of iterations for training")
    args = parser.parse_args()

    df_corrected = pd.read_csv(
        os.path.join(OUTPUT_DIR, f"matching_{args.base_model.replace('/', '_')}-corrected.csv")
    )
    df_train = build_training_dataset(df_corrected)
    log(df_train["label"].value_counts())

    finetune_setfit(df_train, args.base_model, args.output_name, args.test_size, args.num_epochs, args.batch_size, args.num_iterations)

if __name__ == "__main__":
    main()