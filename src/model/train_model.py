# src/model/train_model.py
import os
import joblib
import pandas as pd

# Importas la función get_model() del archivo model_definition.py
from model_definition import get_model

PROCESSED_DATA_DIR = './data/processed/'
MODEL_DIR = './models/'

def load_processed_data():
    """Carga los datos procesados."""
    X_path = os.path.join(PROCESSED_DATA_DIR, 'train_features.csv')
    y_path = os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv')
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)['SalePrice']
    return X, y

def train_and_save_model(X, y):
    """Entrena el modelo y guarda el modelo entrenado."""
    # Usamos tu función get_model() para instanciar el modelo
    model = get_model()
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'trained_model.pkl'))
    print("Modelo entrenado y guardado en trained_model.pkl")

def main():
    """Entrena y guarda el modelo."""
    X, y = load_processed_data()
    train_and_save_model(X, y)

if __name__ == '__main__':
    main()