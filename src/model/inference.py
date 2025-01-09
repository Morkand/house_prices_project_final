import os
import pandas as pd
import joblib

# Paths
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

def load_model_and_preprocessor():
    """
    Carga el modelo entrenado y el preprocesador desde MODEL_DIR.
    Estos archivos se generan tras el entrenamiento (trained_model.pkl y preprocessor.pkl).
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
    Aplica el preprocesador (ya entrenado) al conjunto de prueba.
    Asume que test_data incluye las mismas columnas que se usaron para
    entrenar el pipeline (excepto 'SalePrice' y 'Id', que no están en test).
    """
    # Se elimina 'Id' para quedarse solo con las características
    X_test = test_data.drop(columns=['Id'])

    # Usar transform (NO fit) para aplicar exactamente las mismas
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

    # 3. Preprocesar el conjunto de prueba
    test_features_processed = preprocess_test_data(test_data, preprocessor)

    # 4. Generar predicciones con el modelo
    predictions = model.predict(test_features_processed)

    # 5. Guardar predicciones (asociadas al Id original)
    save_predictions(predictions, test_data)

if __name__ == '__main__':
    main()
