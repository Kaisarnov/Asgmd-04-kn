import os
import joblib
from src.train import train_model
from src.preprocess import create_preprocessor
from src.ingest_data import load_data


def run_pipeline():

   df = load_data("data/train.csv")

   X_train, X_test, y_train, y_test, preprocessor = create_preprocessor(df)

   model = train_model(X_train, y_train, preprocessor)

   return model


if __name__ == "__main__":

    os.makedirs("model", exist_ok=True)

    model = run_pipeline()

    joblib.dump(model, "model/model.pkl")

    print("Model saved successfully")