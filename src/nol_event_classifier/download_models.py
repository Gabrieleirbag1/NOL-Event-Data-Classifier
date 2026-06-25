#download locally models to models dir
import os


MODELS = [
    # "dangvantuan/sentence-camembert-large",
    "paraphrase-multilingual-mpnet-base-v2",
]

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","models"))

def download_models(models):
    from sentence_transformers import SentenceTransformer
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in models:
        print(f"Downloading model: {model_name}")
        model = SentenceTransformer(model_name)
        save_path = os.path.join(OUTPUT_DIR, model_name.replace("/", "_"))
        model.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    download_models(MODELS)