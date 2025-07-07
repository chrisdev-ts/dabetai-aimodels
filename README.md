# dabetai

## Descripción

dabetai es una plataforma integral para el manejo de la diabetes tipo 1 que combina una aplicación web para profesionales de la salud y una aplicación móvil para pacientes. El sistema utiliza modelos de inteligencia artificial entrenados con datos clínicos y métricas de monitoreo continuo de glucosa (CGM) para predecir complicaciones diabéticas específicas.

### Complicaciones soportadas

- **Retinopatía diabética**: Modelo optimizado con 13 características específicas
- **Nefropatía diabética**: Modelo con 18 características incluyendo variables renales
- **Neuropatía diabética**: Modelo con 18 características enfocadas en función neurológica
- **Pie diabético**: Modelo con 18 características específicas para complicaciones podológicas

### Enfoque técnico del sistema de IA

El módulo de inteligencia artificial implementa un pipeline completo de machine learning que incluye:

- Preparación automatizada de datasets con características específicas por complicación
- Experimentación comparativa con múltiples algoritmos (Regresión Logística, Random Forest, LightGBM, XGBoost, SVM, AdaBoost)
- Optimización de hiperparámetros mediante búsqueda en grilla
- Finalización y serialización de modelos optimizados
- Generación automática de reportes y visualizaciones

## Estructura del proyecto

```
dabetai/
├── scripts/                     # Scripts de machine learning
│   ├── 01_prepare_datasets.py   # Preparación modular de datasets
│   ├── 02_run_experiments.py    # Experimentación y comparación de modelos
│   └── 03_finalize_model.py     # Finalización de modelos optimizados
├── data/                        # Datos del proyecto
│   ├── raw/                     # Datos clínicos originales (IOBP2)
│   └── processed/               # Datasets procesados por complicación
├── models/                      # Modelos entrenados (.joblib)
├── reports/                     # Reportes y visualizaciones
│   └── figures/                 # Gráficos por complicación
└── requirements.txt             # Dependencias del proyecto
```

### Scripts principales

- **`01_prepare_datasets.py`**: Generación modular de datasets con características específicas por complicación
- **`02_run_experiments.py`**: Experimentación automatizada con múltiples algoritmos y optimización de hiperparámetros
- **`03_finalize_model.py`**: Entrenamiento final y serialización de modelos optimizados

## Características del sistema

### 🔧 Sistema modular

- **Configuración centralizada**: Definición única de algoritmos y complicaciones
- **Selección automática de características**: Cada complicación utiliza sus características más relevantes
- **Pipeline flexible**: Escalado automático según el algoritmo seleccionado

### 📊 Características por complicación

- **Retinopatía**: 13 características (demográficas, metabólicas, glucémicas)
- **Otras complicaciones**: 18 características (incluye variables de comportamiento e insulina)

### 🤖 Algoritmos disponibles

- **Regresión Logística** (con escalado automático)
- **Random Forest**
- **LightGBM**
- **XGBoost**
- **SVM con kernel RBF** (con escalado automático)
- **AdaBoost**

### 📈 Pipeline de ML completo

1. **Imputación de valores faltantes** (mediana)
2. **Escalado de características** (cuando es necesario)
3. **Balanceo de clases** (SMOTE)
4. **Optimización de hiperparámetros** (GridSearchCV)
5. **Validación cruzada estratificada**
6. **Generación de reportes y visualizaciones**

## Requisitos

- Python 3.8 o superior
- Bibliotecas principales:
  ```
  pandas>=1.5.0
  numpy>=1.21.0
  scikit-learn>=1.1.0
  imbalanced-learn>=0.9.0
  lightgbm>=3.3.0
  xgboost>=1.6.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  joblib>=1.1.0
  ```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Guía de uso

### 1. Preparación de datasets

Genera datasets específicos para cada complicación:

```bash
cd scripts
python 01_prepare_datasets.py
```

**Opciones disponibles:**

- Generar todos los datasets (por defecto)
- Generar dataset específico: `generar_dataset_especifico("retinopathy")`
- Ver configuraciones: `mostrar_resumen_configuraciones()`

### 2. Experimentación con modelos

Ejecuta experimentos comparativos automáticos:

```bash
python 02_run_experiments.py
```

**Genera automáticamente:**

- Comparación de todos los algoritmos disponibles
- Optimización de hiperparámetros
- Reportes comparativos y del modelo ganador
- Visualizaciones adaptativas (ROC, importancia de características, matrices de confusión)

### 3. Finalización de modelos

Entrena y guarda los modelos finales optimizados:

```bash
python 03_finalize_model.py
```

**Configuración actual (modelos ganadores):**

- **Retinopatía**: AdaBoost (learning_rate=0.1, n_estimators=50)
- **Nefropatía**: Regresión Logística (C=0.01)
- **Neuropatía**: Regresión Logística (C=0.01)
- **Pie Diabético**: Regresión Logística (C=0.01)

## Outputs generados

### Datasets procesados

```
data/processed/
├── retinopathy_model_dataset.csv    # 13 características
├── nephropathy_model_dataset.csv    # 18 características
├── neuropathy_model_dataset.csv     # 18 características
└── diabetic_foot_model_dataset.csv  # 18 características
```

### Modelos finalizados

```
models/
├── retinopathy_model.joblib
├── nephropathy_model.joblib
├── neuropathy_model.joblib
└── diabetic_foot_model.joblib
```

### Reportes y visualizaciones

```
reports/figures/[Complicación]/
├── COMPARATIVE_cv_metrics.png       # Métricas de validación cruzada
├── COMPARATIVE_roc_curves.png       # Curvas ROC comparativas
├── COMPARATIVE_table.png           # Tabla comparativa de modelos
├── FINAL_classification_report.png # Reporte del modelo final
├── FINAL_confusion_matrix.png      # Matriz de confusión
├── FINAL_feature_importance.png    # Importancia de características
└── FINAL_roc_curve.png            # Curva ROC del modelo final
```

## Datos requeridos

El sistema requiere datos del estudio **IOBP2 (In Control)** - un ensayo clínico randomizado de diabetes tipo 1. Los datos deben colocarse en `data/raw/datatables/` con los siguientes archivos:

### Archivos necesarios

- `IOBP2DeviceCGM.txt` - Datos de monitoreo continuo de glucosa
- `IOBP2MedicalCondition.txt` - Condiciones médicas y complicaciones
- `IOBP2DiabScreening.txt` - Datos de screening y demografía
- `IOBP2PtRoster.txt` - Información básica de pacientes
- `IOBP2HeightWeight.txt` - Datos antropométricos
- `IOBP2RandBaseInfo.txt` - Información de insulina
- `IOBP2DiabSocioEcon.txt` - Variables socioeconómicas
- `IOBP2PSHFSAdultNoPart2.txt` - Escalas de miedo a hipoglucemia
- `IOBP2PST1DDS.txt` - Escalas de angustia por diabetes

### Cómo obtener los datos

Los datos del estudio IOBP2 están disponibles públicamente a través de:

- **JAEB Center for Health Research**: https://public.jaeb.org/dataset/579
- **Título del dataset**: "The Insulin-Only Bionic Pancreas Pivotal Trial: Testing the iLet in Adults and Children with Type 1 Diabetes"
- **Registro requerido**: Es necesario completar un formulario con información personal e institucional

### Atribución y uso responsable

**⚠️ IMPORTANTE**: Este dataset requiere atribución específica y cumplimiento de condiciones de uso.

**Consulta `CITATION.md` para información detallada sobre atribución requerida, condiciones de uso y uso responsable de los datos.**

## Notas técnicas

- **Reproducibilidad**: Todos los modelos utilizan `random_state=42`
- **Balanceo de clases**: SMOTE aplicado automáticamente para manejar desbalance
- **Validación**: Validación cruzada estratificada con 5 folds
- **Métricas**: AUC-ROC, precisión, recall, F1-score, especificidad
- **Formato de modelos**: Pipelines completos serializados con joblib

## Documentación adicional

Para información detallada, consulta:

- `GUIA_SISTEMA_MODULARIZADO.md` - Guía completa del sistema modular
- `CITATION.md` - Atribución del dataset y uso responsable de los datos

## Licencia

Este proyecto está bajo la licencia MIT.
