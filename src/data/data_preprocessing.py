import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Paths
RAW_DATA_DIR = './data/raw/'
PROCESSED_DATA_DIR = './data/processed/'
MODEL_DIR = './models/'

def load_data():
    """Carga los datos originales desde la carpeta raw."""
    train_path = os.path.join(RAW_DATA_DIR, 'train.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'test.csv')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data_train(data):
    """
    Preprocesa los datos de entrenamiento:
      - Separa atributos y la variable objetivo.
      - Ajusta (fit) el pipeline y transforma (transform) X_train.
      - Devuelve X_train procesado, y_train y el preprocessor entrenado.
    """
    # Separar características y target
    X = data.drop(columns=['SalePrice', 'Id'])
    y = data['SalePrice']

    # Columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Pipeline para columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Pipeline para columnas numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer para combinar ambas transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Ajustar (fit) y transformar (transform) usando datos de entrenamiento
    X_processed = preprocessor.fit_transform(X)

    # Extraer los nombres de las nuevas columnas (numéricas + OneHot)
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'] \
                                     .get_feature_names_out(categorical_cols)
    feature_names = list(numerical_cols) + list(cat_feature_names)

    # Convertir a DataFrame (ojo que .fit_transform() puede devolver matriz dispersa)
    X_processed_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)

    return X_processed_df, y, preprocessor

def preprocess_data_test(data, preprocessor):
    """
    Preprocesa los datos de prueba usando el preprocessor
    ya entrenado con los datos de entrenamiento.
    """
    # En test no existe 'SalePrice', sólo removemos 'Id'
    X = data.drop(columns=['Id'])

    # Transformar (NO hacer fit) con el preprocesador entrenado
    X_processed = preprocessor.transform(X)

    # Recuperar las columnas asociadas al preprocessor
    # (mismo orden y misma cantidad de columnas que en entrenamiento)
    # Para ello, usamos las columnas definidas internamente:
    #   - preprocessor.transformers_ nos da info de cada pipeline
    categorical_cols = preprocessor.transformers_[1][2]  # índice 1 -> ('cat', ...)
    numerical_cols = preprocessor.transformers_[0][2]    # índice 0 -> ('num', ...)
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'] \
                                     .get_feature_names_out(categorical_cols)
    feature_names = list(numerical_cols) + list(cat_feature_names)

    # Convertir a DataFrame
    X_processed_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)

    return X_processed_df

def save_processed_data(X, y):
    """Guarda en disco los datos procesados de entrenamiento."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    X.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_features.csv'), index=False)
    if y is not None:
        y.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv'), index=False)

def save_test_processed_data(X):
    """Guarda en disco los datos procesados de prueba."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    X.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_features.csv'), index=False)

def save_preprocessor(preprocessor):
    """Guarda el preprocesador entrenado en la carpeta de modelos."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))

def main():
    """Ejecuta el pipeline de preprocesamiento de datos."""
    # 1. Cargar datos
    train_data, test_data = load_data()

    # 2. Preprocesar datos de entrenamiento
    X_train, y_train, preprocessor = preprocess_data_train(train_data)

    # 3. Preprocesar datos de prueba con el preprocesador entrenado
    X_test = preprocess_data_test(test_data, preprocessor)

    # 4. Guardar datos procesados y el preprocesador
    save_processed_data(X_train, y_train)
    save_test_processed_data(X_test)
    save_preprocessor(preprocessor)

if __name__ == '__main__':
    main()
