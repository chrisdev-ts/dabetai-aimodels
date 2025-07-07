# Guía de uso del sistema modularizado de dabetai

## 📋 Descripción general

El sistema dabetai está completamente modularizado para permitir máxima flexibilidad en:

- **Selección de características** por complicación
- **Selección de algoritmos** de machine learning
- **Configuración de hiperparámetros** personalizados
- **Ejecución de experimentos** individuales o masivos

## 🗂️ Estructura del sistema

### Scripts principales

1. **`01_prepare_datasets.py`** - Generación de datasets clínicos
2. **`02_run_experiments.py`** - Ejecución de experimentos comparativos
3. **`03_finalize_model.py`** - Finalización de modelos optimizados

## 🔧 Configuración centralizada

### Complicaciones disponibles

```python
complicaciones = {
    "retinopathy": "Retinopathy_Status",
    "nephropathy": "Nephropathy_Status",
    "neuropathy": "Neuropathy_Status",
    "diabetic_foot": "Diabetic_Foot_Status"
}
```

### Algoritmos disponibles

- **Regresión Logística** (con escalado)
- **Random Forest** (sin escalado)
- **LightGBM** (sin escalado)
- **XGBoost** (sin escalado)
- **SVM (RBF Kernel)** (con escalado)
- **AdaBoost** (sin escalado)

### Características por complicación

- **Retinopatía**: 13 características
- **Nefropatía**: 18 características
- **Neuropatía**: 18 características
- **Pie Diabético**: 18 características

## 🚀 Guía de uso

### 1. Preparación de datasets

```bash
cd scripts
python 01_prepare_datasets.py
```

**Características:**

- Genera datasets específicos para cada complicación
- Configuración centralizada de características
- Resumen detallado de configuraciones
- Validación automática de datos

### 2. Ejecución de experimentos

```bash
python 02_run_experiments.py
```

**Características:**

- Compara múltiples algoritmos automáticamente
- Genera reportes comparativos y del modelo ganador
- Visualizaciones adaptativas según número de características
- Métricas completas de rendimiento

### 3. Finalización de modelos

#### Opción A: Finalizar todos los modelos

```bash
python 03_finalize_model.py
```

#### Opción B: Finalizar un solo modelo

```python
from finalize_model import finalize_single_model

finalize_single_model(
    complication_name="retinopathy",
    model_name="Random Forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 5
    }
)
```

#### Opción C: Finalizar múltiples modelos personalizados

```python
from finalize_model import finalize_multiple_models

custom_models = [
    {
        "complication_name": "retinopathy",
        "model_name": "XGBoost",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
    },
    {
        "complication_name": "nephropathy",
        "model_name": "LightGBM",
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.05
        }
    }
]

finalize_multiple_models(custom_models)
```

## 📊 Outputs y resultados

### Datasets generados

- `data/processed/retinopathy_model_dataset.csv`
- `data/processed/nephropathy_model_dataset.csv`
- `data/processed/neuropathy_model_dataset.csv`
- `data/processed/diabetic_foot_model_dataset.csv`

### Modelos finalizados

- `models/retinopathy_model.joblib`
- `models/nephropathy_model.joblib`
- `models/neuropathy_model.joblib`
- `models/diabetic_foot_model.joblib`

### Reportes y visualizaciones

- `reports/figures/[Complicación]/`
- Gráficos comparativos y del modelo final
- Métricas de rendimiento detalladas
- Matrices de confusión
- Curvas ROC
- Importancia de características
