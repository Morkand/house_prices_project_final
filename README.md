# Proyecto de Regresión de Precios de Casas

Este proyecto predice los precios de casas utilizando el conjunto de datos **Ames Housing**. El pipeline incluye preprocesamiento de datos, entrenamiento del modelo e inferencia para generar predicciones.

---

## **Estructura del Proyecto**

```
project_name/
│
├── data/
│   ├── raw/               # Archivos originales del dataset
│   ├── processed/         # Archivos procesados del dataset
│
├── models/                # Modelos entrenados
│
├── src/                   # Código fuente
│   ├── data/              # Scripts de preprocesamiento de datos
│   ├── model/             # Definición, entrenamiento e inferencia del modelo
│   ├── tests/             # Pruebas unitarias
│
├── pyproject.toml         # Dependencias gestionadas con Poetry
├── .gitignore             # Archivos ignorados por Git
├── README.md              # Documentación del proyecto
```

---

## **Comenzando**

### **Requisitos Previos**

- Python 3.9 o superior.
- Poetry (para la gestión de dependencias).
- Docker (opcional, para contenedores).

### **Instalación**

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Morkand/house_prices_project_final
   ```
2. Accede al directorio del proyecto:
   ```bash
   cd project_name
   ```
3. Instala las dependencias usando Poetry:
   ```bash
   poetry install
   ```

---

## **Uso**

### **1. Preprocesar Datos**

Limpia y prepara los datos para el entrenamiento y la inferencia:

```bash
python src/data/data_preprocessing.py
```

### **2. Entrenar el Modelo**

Entrena el modelo de regresión utilizando los datos preprocesados:

```bash
python src/model/train_model.py
```

### **3. Generar Predicciones**

Usa el modelo entrenado para realizar predicciones sobre los datos de prueba:

```bash
python src/model/inference.py
```

### **4. Pruebas**

Ejecuta las pruebas unitarias del pipeline:

```bash
pytest src/tests
```

---

## **Uso con Docker (Opcional)**

Puedes ejecutar el proyecto en un contenedor Docker para facilitar su implementación:

1. Construye la imagen del contenedor:
   ```bash
   docker build -t house-prices-project .
   ```
2. Ejecuta el contenedor:
   ```bash
   docker run -p 5200:5200 house-prices-project
   ```

Esto ejecutará la aplicación Flask en el puerto `5200`.

---

## **Pruebas Unitarias**

Este proyecto incluye pruebas unitarias para asegurar la calidad del pipeline:

- **Preprocesamiento de datos:** Validación de la limpieza y transformación de los datos.
- **Entrenamiento del modelo:** Verificación de la correcta ejecución del entrenamiento.
- **Inferencia:** Validación del formato y contenido de las predicciones generadas.
- **Integración con Docker:** Ejecución de pruebas dentro del contenedor.

Ejecuta las pruebas con:

```bash
pytest src/tests
```

---

## **Licencia**

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
