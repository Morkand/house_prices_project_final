import pytest
from flask import Flask
from flask.testing import FlaskClient

@pytest.fixture
def app() -> Flask:
    """Crear y configurar una nueva aplicación Flask para cada prueba."""
    app = Flask(__name__)

    # Configurar rutas para pruebas
    app.add_url_rule('/', 'index', lambda: 'House Prices Project')  # Página principal
    app.add_url_rule('/data', 'data', lambda: 'Página de datos')    # Página de datos
    app.add_url_rule('/analysis', 'analysis', lambda: 'Página de análisis')  # Página de análisis

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Devolver un cliente de prueba para la aplicación Flask."""
    return app.test_client()


def test_pagina_principal(client: FlaskClient):
    """Probar la ruta de la página principal."""
    response = client.get('/')
    assert response.status_code == 200
    assert "House Prices Project" in response.data.decode('utf-8')  # Decodificar antes de comparar


def test_pagina_datos(client: FlaskClient):
    """Probar la ruta de la página de datos."""
    response = client.get('/data')
    assert response.status_code == 200
    # Decodificar antes de comparar
    assert "Página de datos" in response.data.decode('utf-8')


def test_pagina_analisis(client: FlaskClient):
    """Probar la ruta de la página de análisis."""
    response = client.get('/analysis')
    assert response.status_code == 200
    # Decodificar antes de comparar
    assert "Página de análisis" in response.data.decode('utf-8')