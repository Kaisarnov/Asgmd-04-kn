import os
import joblib

from ingest_data import load_data
from preprocess import create_preprocessor
from train import train_model
from evaluate import evaluate
from sklearn.model_selection import train_test_split

def run_pipeline():

    df = load_data("data/train.csv")

    X = df.drop("Transported", axis=1)
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = create_preprocessor()

    model = train_model(X_train, y_train, preprocessor)

    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
    os.makedirs("model", exist_ok=True)

    model = os.path.join("model", "model.pkl")
    joblib.dump(model, model)

    print(f"Model saved at {model}")
