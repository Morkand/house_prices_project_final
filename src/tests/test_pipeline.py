import os
import sys
import pytest
import pandas as pd
from joblib import load
from flask import Flask
from flask.testing import FlaskClient
from werkzeug.datastructures import FileStorage

# Agregar el directorio raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Configuración de rutas
MODEL_PATH = './models/trained_model.pkl'
PREPROCESSOR_PATH = './models/preprocessor.pkl'
TEST_DATA_PATH = './data/raw/test.csv'
TRAIN_DATA_PATH = './data/raw/train.csv'
PREDICTIONS_PATH = './data/processed/predictions.csv'

@pytest.fixture
def model_and_preprocessor():
    """Cargar el modelo y el preprocesador para pruebas."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        pytest.fail("Modelo o preprocesador no encontrados.")

    model = load(MODEL_PATH)
    preprocessor = load(PREPROCESSOR_PATH)
    return model, preprocessor

def test_preprocessor_valid_data(model_and_preprocessor):
    """Probar que el preprocesador transforma correctamente datos válidos."""
    _, preprocessor = model_and_preprocessor

    # Usa datos de prueba con todas las columnas necesarias
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    required_columns = preprocessor.feature_names_in_
    test_data = train_data[required_columns].iloc[:1]  # Usa solo una fila válida

    transformed_data = preprocessor.transform(test_data)
    assert transformed_data.shape[1] > 0, "La transformación del preprocesador falló."

    # Mensaje de éxito
    print("✔ Prueba exitosa: Preprocesador transformó los datos correctamente.")

def test_model_prediction_output(model_and_preprocessor):
    """Validar la estructura del archivo de predicciones generado."""
    from src.model.inference import main as inference_main
    inference_main()
    assert os.path.exists(PREDICTIONS_PATH), "El archivo de predicciones no fue generado."

    predictions = pd.read_csv(PREDICTIONS_PATH)
    assert 'Id' in predictions.columns, "La columna 'Id' no está presente en las predicciones."
    assert 'Predicted' in predictions.columns, "La columna 'Predicted' no está presente en las predicciones."

    # Mensaje de éxito
    print("✔ Prueba exitosa: Predicciones generadas correctamente con columnas 'Id' y 'Predicted'.")

@pytest.fixture
def app():
    """Cargar la aplicación Flask real desde src.app."""
    from src.app import app
    return app

@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Devolver un cliente de prueba para la aplicación Flask."""
    return app.test_client()

def test_pagina_principal(client: FlaskClient):
    """Probar la ruta de la página principal."""
    response = client.get('/')
    assert response.status_code == 200
    assert "Proyecto de Predicción de Precios de Casas" in response.data.decode('utf-8')

    # Mensaje de éxito
    print("✔ Prueba exitosa: Ruta principal funciona correctamente.")
