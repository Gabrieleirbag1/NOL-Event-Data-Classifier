import optuna
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# 1. Charger ton dataset
df = pd.read_csv("output/clusters_paraphrase-multilingual-mpnet-base-v2.csv")
df = df[df["cluster_id"] != -1]  # retire les outliers non labelisés

dataset = Dataset.from_pandas(df[["event", "cluster_id"]].rename(
    columns={"event": "text", "cluster_id": "label"}
))
split = dataset.train_test_split(test_size=0.2, seed=42)

# 2. Tester les 2 modèles
MODELS_TO_FINETUNE = [
    "paraphrase-multilingual-mpnet-base-v2",
    "dangvantuan/sentence-camembert-large",
]

results = {}
for model_name in MODELS_TO_FINETUNE:
    print(f"\nFine-tuning : {model_name}")

    model = SetFitModel.from_pretrained(model_name)
    args = TrainingArguments(
        num_epochs=3,
        batch_size=16,
        num_iterations=20,      # paires contrastives générées
    )
    def model_init(params):
        params = params or {}
        model = SetFitModel.from_pretrained(model_name)
        model.labels = list(set(split["train"]["label"])) # Ensure labels are set before model_init returns depending on version
        return model

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        metric="accuracy",
    )
    
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_epochs": trial.suggest_int("num_epochs", 1, 5),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_iterations": trial.suggest_categorical("num_iterations", [10, 20]),
            "seed": trial.suggest_int("seed", 1, 40),
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        }

    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=10)
    print(f"Best run: {best_run}")
    
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()

    metrics = trainer.evaluate()
    results[model_name] = metrics
    print(f"  → Accuracy: {metrics['accuracy']:.3f}")

    model.save_pretrained(f"output/setfit_{model_name.replace('/', '_')}")

# 3. Comparer
print("\n=== Comparaison des modèles ===")
for model_name, metrics in results.items():
    print(f"{model_name}: accuracy={metrics['accuracy']:.3f}")

from sklearn.metrics import classification_report

# Pour chaque modèle fine-tuné
model = SetFitModel.from_pretrained("output/setfit_dangvantuan_sentence-camembert-large")
preds = model.predict(split["test"]["text"])

print(classification_report(
    split["test"]["label"],
    preds,
))