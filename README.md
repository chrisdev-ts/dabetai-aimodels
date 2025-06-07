# DabetAI

## Descripción

DabetAI es una plataforma diseñada para mejorar el manejo de la diabetes mediante la predicción de complicaciones específicas. Actualmente, el proyecto puede predecir retinopatía, pero la idea es escalarlo para incluir neuropatía, nefropatía y pie diabético.

### Aplicación móvil

La aplicación móvil está enfocada en el paciente con diabetes, permitiéndole conectar sus dispositivos de monitoreo/wearables para recibir predicciones sobre las complicaciones mencionadas.

### Aplicación web

La aplicación web está diseñada para que el médico pueda dar seguimiento general a la enfermedad del paciente.

## Estructura del Proyecto

- **`scripts/`**: Contiene los scripts Python para preparar los datos, entrenar modelos y finalizar el modelo.
  - `01_prepare_dataset.py`: Prepara los datos crudos para el entrenamiento.
  - `02_train_evaluate_models.py`: Entrena y evalúa modelos de regresión logística, Random Forest y LightGBM.
  - `03_finalize_model.py`: Entrena el modelo final y lo guarda.
- **`data/`**: Contiene los datasets.
  - `raw/`: Archivos de datos originales.
  - `processed/`: Archivos de datos listos para el modelo.
- **`models/`**: Contiene los modelos entrenados.
  - `rf_prediction_model.joblib`: Modelo entrenado con Random Forest.
  - `rl_prediction_model.joblib`: Modelo entrenado con regresión logística.
- **`README.md`**: Documentación del proyecto.

## Funcionalidades

1. **Carga y preprocesamiento de datos**: Limpieza inicial y conversión de formatos.
2. **Generación de variable objetivo**: Identificación de pacientes con retinopatía.
3. **Entrenamiento de modelos**: Entrenamiento de regresión logística, Random Forest y LightGBM.
4. **Evaluación de modelos**: Comparación de métricas de rendimiento.
5. **Finalización del modelo**: Entrenamiento del modelo final y guardado.

## Requisitos

- Python 3.8 o superior
- Bibliotecas necesarias:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imblearn`
  - `lightgbm`
  - `joblib`

## Uso

1. **Preparar los datos**:
   Ejecuta el script para preparar los datos crudos:

   ```bash
   python scripts/01_prepare_dataset.py
   ```

2. **Entrenar y evaluar modelos**:
   Ejecuta el script para entrenar y evaluar los modelos:

   ```bash
   python scripts/02_train_evaluate_models.py
   ```

3. **Finalizar el modelo**:
   Ejecuta el script para entrenar el modelo final y guardarlo:
   ```bash
   python scripts/03_finalize_model.py
   ```

## Notas

- Los archivos de datos están excluidos del repositorio mediante `.gitignore`.
- Asegúrate de que los archivos de datos sigan el formato esperado para evitar errores durante la ejecución.

## Licencia

Este proyecto está bajo la licencia MIT.
