import os
import pandas as pd
import numpy as np
import joblib

# ====== RUTAS ======
RAW_DATA_DIR = './data/raw/'
PROCESSED_DATA_DIR = './data/processed/'
MODEL_DIR = './models/'

def load_test_data():
    """
    Carga los datos de prueba desde la carpeta raw.
    Asume que existe un archivo test.csv que contiene
    todas las columnas necesarias, incluida la columna 'Id'.
    """
    test_path = os.path.join(RAW_DATA_DIR, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró el archivo de prueba en {test_path}")

    test_data = pd.read_csv(test_path)

    # Verificar que exista la columna 'Id', dado que la usas para asociar las predicciones
    if 'Id' not in test_data.columns:
        raise ValueError("El archivo de prueba debe contener una columna 'Id'.")

    return test_data


def clean_data(data):
    """
    Realiza la misma limpieza que en el script de entrenamiento:
    - Reemplaza valores 'NA' que significan ausencia real (None) en
      columnas específicas de sótano, chimenea, etc.
    - Convierte 'MSSubClass' a string (categórica).
    - Maneja GarageYrBlt cuando no hay garage, etc.
    """
    none_cols = [
        'Alley',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'FireplaceQu',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'PoolQC',
        'Fence',
        'MiscFeature'
    ]
    for c in none_cols:
        data[c] = data[c].replace("NA", np.nan)
        data[c] = data[c].fillna("None")

    # Si no hay garage, a menudo 'GarageYrBlt' está vacío.
    if 'GarageYrBlt' in data.columns:
        data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

    # MSSubClass es numérico pero representa categorías
    if 'MSSubClass' in data.columns:
        data['MSSubClass'] = data['MSSubClass'].astype(str)

    return data


def load_model_and_preprocessor():
    """
    Carga el modelo entrenado y el preprocesador desde MODEL_DIR.
    Estos archivos se generan tras el entrenamiento
    (p. ej. 'trained_model.pkl' y 'preprocessor.pkl').
    """
    model_path = os.path.join(MODEL_DIR, 'trained_model.pkl')
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}.")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"No se encontró el preprocesador en {preprocessor_path}.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def preprocess_test_data(test_data, preprocessor):
    """
    Limpia y transforma el conjunto de prueba usando el preprocesador entrenado.
    1. Realiza la misma limpieza que en training (clean_data).
    2. Elimina la columna 'Id'.
    3. Aplica preprocessor.transform() para obtener el feature matrix final.
    """
    # 1) Limpieza de datos (igual que en entrenamiento)
    test_data = clean_data(test_data)

    # 2) Eliminar 'Id' para quedarse con solo las características
    X_test = test_data.drop(columns=['Id'])

    # 3) Usar transform (NO fit) para aplicar exactamente las mismas
    # transformaciones que se aprendieron en entrenamiento
    X_test_processed = preprocessor.transform(X_test)

    return X_test_processed


def save_predictions(predictions, test_data):
    """
    Guarda las predicciones en la carpeta processed,
    junto con la columna 'Id' para identificar cada registro.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    predictions_path = os.path.join(PROCESSED_DATA_DIR, 'predictions.csv')

    # Crear DataFrame con 'Id' original y la columna 'Predicted'
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Predicted': predictions
    })

    output.to_csv(predictions_path, index=False)
    print(f"Predicciones guardadas en: {predictions_path}")


def main():
    # 1. Cargar datos de prueba
    test_data = load_test_data()

    # 2. Cargar el modelo y el preprocesador
    model, preprocessor = load_model_and_preprocessor()

    # 3. Preprocesar el conjunto de prueba (limpieza + transform)
    X_test_processed = preprocess_test_data(test_data, preprocessor)

    # 4. Generar predicciones con el modelo
    predictions = model.predict(X_test_processed)

    # 5. Guardar predicciones (asociadas al 'Id' original)
    save_predictions(predictions, test_data)


if __name__ == '__main__':
    main()