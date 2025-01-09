
# House Prices Regression Project

This project predicts house prices using the Ames Housing dataset. The pipeline includes data preprocessing, model training, and inference.

## Project Structure
```
project_name/
│
├── data/
│   ├── raw/               # Original dataset files
│   ├── processed/         # Processed dataset files
│
├── models/                # Trained models
│
├── src/                   # Source code
│   ├── data/              # Data preprocessing scripts
│   ├── model/             # Model definition, training, and inference scripts
│
├── pyproject.toml         # Poetry dependencies
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Poetry (for dependency management)

### Installation
1. Clone this repository.
2. Navigate to the project directory.
3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

### Usage

#### 1. Preprocess Data
Run the data preprocessing script to clean and prepare the data:
```bash
python src/data/data_preprocessing.py
```

#### 2. Train Model
Train the regression model using preprocessed data:
```bash
python src/model/train_model.py
```

#### 3. Make Predictions
Use the trained model to make predictions on test data:
```bash
python src/model/inference.py
```

### Docker (Optional)
Build and run the project in a Docker container:
```bash
docker build -t house-prices-project .
docker run -p 8000:8000 house-prices-project
```

### License
This project is licensed under the MIT License.
