from flask import Flask, render_template, request, redirect, url_for, send_file, session
import pandas as pd
import numpy as np
import os
import joblib
import time
# Importamos más métricas de scikit-learn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necesario para usar sesión
UPLOAD_FOLDER = '/uploads/'
PROCESSED_FOLDER = '/data/processed/'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Crear carpetas necesarias si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Directorios y archivos
MODEL_DIR = './models/'
MODEL_FILE = 'trained_model.pkl'
PREPROCESSOR_FILE = 'preprocessor.pkl'

# Verificar existencia de modelo y preprocesador
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
preprocessor_path = os.path.join(MODEL_DIR, PREPROCESSOR_FILE)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en {model_path}.")

if not os.path.exists(preprocessor_path):
    raise FileNotFoundError(f"No se encontró el preprocesador en {preprocessor_path}.")

# Cargar modelo y preprocesador
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(base_name, extension='csv'):
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}.{extension}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error="ERROR: No se subió ningún archivo.")
        file = request.files['file']

        if file.filename == '' or not allowed_file(file.filename):
            return render_template('predict.html', error="ERROR: El archivo debe ser CSV o Excel.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Leer el archivo cargado
        try:
            if filename.endswith('.csv'):
                uploaded_data = pd.read_csv(filepath)
            elif filename.endswith('.xlsx'):
                uploaded_data = pd.read_excel(filepath)
        except Exception as e:
            return render_template('predict.html', error=f"ERROR: No se pudo procesar el archivo. Detalles: {e}")

        # Validar columnas mínimas necesarias
        raw_columns = list(preprocessor.feature_names_in_)
        missing_columns = set(raw_columns) - set(uploaded_data.columns)
        if missing_columns:
            return render_template(
                'predict.html',
                error=f"ERROR: El archivo cargado no contiene las columnas necesarias: {', '.join(missing_columns)}"
            )

        # Verificar si el archivo tiene la columna 'SalePrice'
        has_sale_price = 'SalePrice' in uploaded_data.columns

        # Preprocesar los datos (eliminar 'Id' y 'SalePrice' si están presentes)
        try:
            features = uploaded_data.drop(columns=['Id', 'SalePrice'], errors='ignore')
            features_transformed = preprocessor.transform(features)
        except Exception as e:
            return render_template('predict.html', error=f"ERROR: Falló la transformación de los datos. Detalles: {e}")

        # Generar predicciones
        predictions = model.predict(features_transformed)

        # Guardar predicciones
        unique_filename = generate_unique_filename("predictions")
        output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], unique_filename)
        uploaded_data['Predicted'] = predictions
        uploaded_data.to_csv(output_filepath, index=False)

        # Si incluye 'SalePrice', almacenar el archivo en la sesión para análisis
        if has_sale_price:
            session['analysis_file'] = unique_filename

        return render_template(
            'predict.html',
            success=True,
            download_url=url_for('download_predictions', filename=unique_filename),
            analysis_available=has_sale_price
        )

    return render_template('predict.html')

@app.route('/download_predictions/<filename>', methods=['GET'])
def download_predictions(filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "ERROR: No se encontró el archivo de predicciones.", 404
    return send_file(filepath, as_attachment=True)

@app.route('/analysis', methods=['GET'])
def analysis():
    # Verificar que se proporcionó un archivo de análisis
    analysis_file = session.get('analysis_file')
    if not analysis_file:
        return "ERROR: No se encontró un archivo de análisis. Por favor, realice una predicción primero."

    analysis_filepath = os.path.join(app.config['PROCESSED_FOLDER'], analysis_file)
    if not os.path.exists(analysis_filepath):
        return "ERROR: El archivo de análisis no está disponible."

    predictions_df = pd.read_csv(analysis_filepath)

    if 'SalePrice' in predictions_df.columns:
        # Métricas existentes
        rmse = np.sqrt(mean_squared_error(predictions_df['SalePrice'], predictions_df['Predicted']))
        r2 = r2_score(predictions_df['SalePrice'], predictions_df['Predicted'])

        # Métricas adicionales
        mae = mean_absolute_error(predictions_df['SalePrice'], predictions_df['Predicted'])

        # MAPE puede dar error si SalePrice = 0 en algún registro
        # Manejarlo con try-except si es posible que haya valores de 0
        try:
            mape = mean_absolute_percentage_error(predictions_df['SalePrice'], predictions_df['Predicted'])
        except ZeroDivisionError:
            mape = None

        # Gráficos
        # 1) Comparación de SalePrice vs Predicted
        comparison_plot = px.scatter(
            predictions_df,
            x='SalePrice',
            y='Predicted',
            title='Comparación: Predicciones vs. Precio Real',
            labels={'SalePrice': 'Precio Real ($)', 'Predicted': 'Precio Predicho ($)'},
            template='plotly_dark'
        )
        comparison_html = comparison_plot.to_html(full_html=False)

        # 2) Histograma de Errores
        predictions_df['Error'] = predictions_df['SalePrice'] - predictions_df['Predicted']
        error_plot = px.histogram(
            predictions_df,
            x='Error',
            nbins=50,  # Ajustar la granularidad
            title='Histograma de Errores (SalePrice - Predicted)',
            template='plotly_dark'
        )
        error_html = error_plot.to_html(full_html=False)

    else:
        rmse, r2, mae, mape = None, None, None, None
        comparison_html = None
        error_html = None

    # Gráfico de predicciones (Id vs. Predicted)
    # (si no hay Id en DF, no pasa nada, px.scatter no lo graficará correctamente)
    actual_vs_predicted = px.scatter(
        predictions_df,
        x='Id' if 'Id' in predictions_df.columns else predictions_df.index,
        y='Predicted',
        title='Predicciones de Precios',
        labels={'x': 'ID de la Casa', 'Predicted': 'Precio Predicho ($)'},
        template='plotly_dark'
    )
    actual_vs_predicted_html = actual_vs_predicted.to_html(full_html=False)

    return render_template(
        'analysis.html',
        actual_vs_predicted=actual_vs_predicted_html,
        comparison_graph=comparison_html,
        error_hist=error_html,
        rmse=rmse,
        r2=r2,
        mae=mae,
        mape=mape
    )

@app.route('/data', methods=['GET'])
def data():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 100))

    TRAIN_DATA = './data/processed/train_features.csv'
    TEST_DATA = './data/processed/test_features.csv'
    PREDICTIONS = './data/processed/predictions.csv'

    if not os.path.exists(TRAIN_DATA):
        return f"ERROR: no se encontró {TRAIN_DATA}."
    if not os.path.exists(TEST_DATA):
        return f"ERROR: no se encontró {TEST_DATA}."
    if not os.path.exists(PREDICTIONS):
        return f"ERROR: no se encontró {PREDICTIONS}."

    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)
    predictions = pd.read_csv(PREDICTIONS)

    start = (page - 1) * limit
    end = start + limit

    train_data_paginated = train_data.iloc[start:end]
    test_data_paginated = test_data.iloc[start:end]
    predictions_paginated = predictions.iloc[start:end]

    return render_template(
        'data.html',
        train_data=train_data_paginated.to_html(classes='table table-striped table-bordered', index=False),
        test_data=test_data_paginated.to_html(classes='table table-striped table-bordered', index=False),
        predictions=predictions_paginated.to_html(classes='table table-striped table-bordered', index=False),
        page=page,
        limit=limit,
        total_train=train_data.shape[0],
        total_test=test_data.shape[0],
        total_predictions=predictions.shape[0]
    )

if __name__ == '__main__':
    app.run(debug=True, port=5200, host='0.0.0.0')