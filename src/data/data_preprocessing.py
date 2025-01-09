import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ====== RUTAS ======
RAW_DATA_DIR = './data/raw/'
PROCESSED_DATA_DIR = './data/processed/'
MODEL_DIR = './models/'

# =====================================================
# 1. FUNCIONES DE CARGA Y LIMPIEZA BÁSICA
# =====================================================

def load_data():
    """Carga los datos originales desde la carpeta raw."""
    train_path = os.path.join(RAW_DATA_DIR, 'train.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'test.csv')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def clean_data(data):
    """
    Reemplaza los valores 'NA' (str) en columnas donde 'NA' realmente
    significa 'None' o 'No existe'. También llena con 'None' si hay
    valores perdidos reales (NaN).
    """

    # Estas columnas usan 'NA'/'None' para indicar que no hay sótano, garage, etc.
    none_cols = [
        'Alley',         # No alley access
        'BsmtQual',      # No Basement
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'FireplaceQu',   # No Fireplace
        'GarageType',    # No Garage
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'PoolQC',        # No Pool
        'Fence',         # No Fence
        'MiscFeature'    # No Misc Feature
    ]
    for c in none_cols:
        data[c] = data[c].replace("NA", np.nan)    # Si hay "NA" como string, pásalo a NaN
        data[c] = data[c].fillna("None")           # Rellena NaN con "None"

    # Ejemplo: si no hay garage, a menudo 'GarageYrBlt' está vacío.
    # Podemos ponerlo a 0 (o a alguna otra convención).
    if 'GarageYrBlt' in data.columns:
        data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

    # MSSubClass es numérico pero representa categorías
    # Convertirlo a string para que sea tratado como categoría
    if 'MSSubClass' in data.columns:
        data['MSSubClass'] = data['MSSubClass'].astype(str)

    return data


# =====================================================
# 2. CONFIGURACIÓN DE COLUMNAS (NUM, ORD, NOM)
# =====================================================

# Columnas que son ordinales y *además* tienen categorías definidas.
# Cada clave es el nombre de la columna, y el valor es la lista
# de categorías en orden ascendente.
# OJO: se asume que "None" es la categoría más baja si corresponde.
ORDINAL_MAPS = {
    'ExterQual':    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond':    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual':     ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond':     ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'HeatingQC':    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual':  ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'FireplaceQu':  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageQual':   ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond':   ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PoolQC':       ['None', 'Fa', 'TA', 'Gd', 'Ex']
}

# =====================================================
# 3. FUNCIONES DE PREPROCESAMIENTO
# =====================================================

def preprocess_data_train(data):
    """
    Preprocesa los datos de entrenamiento:
      - Limpia valores "NA"/"None" en columnas que así lo requieran.
      - Separa atributos y la variable objetivo.
      - Define la lógica de transformación para num, ordinal y nominal.
      - Ajusta (fit) el pipeline y transforma (transform) X_train.
      - Devuelve X_train procesado, y_train y el preprocessor entrenado.
    """
    # ----------------------
    # LIMPIEZA ESPECÍFICA
    # ----------------------
    data = clean_data(data)

    # Separar características (X) y target (y)
    X = data.drop(columns=['SalePrice', 'Id'])
    y = data['SalePrice']

    # -------------------------------------------------
    # Definir listas de columnas
    # -------------------------------------------------
    # 1) Columnas ordinales (existentes en el dataset)
    ordinal_cols = [col for col in ORDINAL_MAPS.keys() if col in X.columns]

    # 2) Columnas categóricas nominales:
    #    - Serán todas las columnas de tipo 'object'
    #      que *NO* estén en ordinal_cols
    #    - Notar que MSSubClass es 'object' ahora,
    #      y no está en ORDINAL_MAPS, por tanto irá aquí.
    all_object_cols = X.select_dtypes(include=['object']).columns.tolist()
    nominal_cols = list(set(all_object_cols) - set(ordinal_cols))

    # 3) Columnas numéricas:
    #    - Cogemos las que sean int/float
    #    - Excluimos el target y cualquier otra que
    #      estemos tratando como categórica
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # -------------------------------------------------
    # DEFINICIÓN DE PIPELINES POR TIPO DE COLUMNA
    # -------------------------------------------------

    # a) Numérico
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # b) Ordinal
    #    Construimos el OrdinalEncoder con la lista de categorías
    #    en el orden correcto para cada columna
    ordinal_categories = [ORDINAL_MAPS[col] for col in ordinal_cols]
    ordinal_transformer = Pipeline(steps=[
        # Si llegara un NaN real, lo imputamos con 'None' (o 'Po' si procede).
        # Aquí usamos most_frequent por simplicidad (podrías definir algo más personalizado).
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_enc', OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1  # en caso de que aparezca algo fuera de lo definido
        ))
    ])

    # c) Nominal (One-Hot)
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # -------------------------------------------------
    # Unir todo en un ColumnTransformer
    # -------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols)
        ]
    )

    # -------------------------------------------------
    # Ajustar (fit) y transformar (transform) con datos de entrenamiento
    # -------------------------------------------------
    X_processed = preprocessor.fit_transform(X)

    # Obtener nombres de columnas resultantes
    # 1) num -> numeric_cols
    # 2) ord -> ordinal_cols (sin expandir, 1D por columna)
    # 3) nom -> one-hot expandido
    #    Para nominal, recuperamos sus nombres de la etapa 'onehot'
    onehot = preprocessor.named_transformers_['nom']['onehot']
    # ordinal no se expande, cada col sigue siendo 1 dimensión
    new_ordinal_names = ordinal_cols
    # nominal se expande en dummies
    new_nominal_names = onehot.get_feature_names_out(nominal_cols)

    feature_names = numeric_cols + new_ordinal_names + list(new_nominal_names)

    # Convertir a DataFrame (ojo que .fit_transform() puede devolver matriz dispersa)
    if hasattr(X_processed, 'toarray'):
        X_processed_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)
    else:
        # Si fuese un np.ndarray denso
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    return X_processed_df, y, preprocessor


def preprocess_data_test(data, preprocessor):
    """
    Preprocesa los datos de prueba usando el preprocessor
    ya entrenado con los datos de entrenamiento.
    """
    # Limpiar datos igual que en train
    data = clean_data(data)

    # En test no hay 'SalePrice', sólo removemos 'Id' (si existe)
    if 'Id' in data.columns:
        X = data.drop(columns=['Id'])
    else:
        X = data

    # Transformar (NO hacer fit) con el preprocesador entrenado
    X_processed = preprocessor.transform(X)

    # Recuperar las columnas asociadas al preprocessor:
    # Recordar que las definimos en el train con numeric_cols, ordinal_cols,
    # y nominal_cols (OneHot). Podemos extraerlas igual que en train.
    numeric_cols = preprocessor.transformers_[0][2]  # ('num', ...)
    ordinal_cols = preprocessor.transformers_[1][2]  # ('ord', ...)
    nominal_cols = preprocessor.transformers_[2][2]  # ('nom', ...)

    # Nombres ordinales (mismos que train)
    new_ordinal_names = ordinal_cols

    # Nombres nominales (OneHot)
    onehot = preprocessor.named_transformers_['nom']['onehot']
    new_nominal_names = onehot.get_feature_names_out(nominal_cols)

    feature_names = list(numeric_cols) + list(new_ordinal_names) + list(new_nominal_names)

    # Convertir a DataFrame
    if hasattr(X_processed, 'toarray'):
        X_processed_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)
    else:
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    return X_processed_df

# =====================================================
# 4. FUNCIONES PARA GUARDAR RESULTADOS
# =====================================================

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


# =====================================================
# 5. MAIN
# =====================================================

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