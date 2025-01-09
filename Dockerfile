# Imagen base de Python 3.9 slim
FROM python:3.9-slim

# Configuramos el directorio de trabajo
WORKDIR /app

# Copiar todos los archivos del proyecto dentro del contenedor
COPY . .

# Crear directorios necesarios dentro del contenedor
RUN mkdir -p data/processed

# Actualizar y instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python (actualización de pip, setuptools y wheel)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar Poetry y pytest para la gestión de dependencias y pruebas
RUN pip install poetry pytest

# Instalar las dependencias del proyecto utilizando Poetry
RUN poetry install --no-root

# Ejecutar las pruebas para verificar el pipeline
RUN poetry run pytest src/tests/test_pipeline.py

# Ejecutar el preprocesamiento, entrenamiento e inferencia como parte del pipeline
RUN poetry run python src/data/data_preprocessing.py
RUN poetry run python src/model/train_model.py
RUN poetry run python src/model/inference.py

# Exponer el puerto 5200 para Flask
EXPOSE 5200

# Ajustar permisos de los archivos
RUN chmod -R 755 /app

# Comando predeterminado para ejecutar la aplicación Flask
CMD ["poetry", "run", "python", "src/app.py"]
