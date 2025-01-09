
from sklearn.ensemble import RandomForestRegressor

def get_model():
    """Define y retorna un modelo de Random Forest para regresión."""
    model = RandomForestRegressor(
        n_estimators=100,  # Número de árboles
        random_state=42,   # Para reproducibilidad
        n_jobs=-1          # Usar todos los núcleos disponibles
    )
    return model
