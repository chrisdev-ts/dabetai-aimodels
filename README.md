# DabetAI

## Descripción

DabetAI es un proyecto de ingeniería de características diseñado para la predicción de retinopatía en pacientes con diabetes. Utiliza datos clínicos y demográficos para generar un conjunto de datos procesado que puede ser utilizado en modelos de aprendizaje automático.

## Estructura del Proyecto

- `feature-engineering.py`: Script principal que realiza la carga, limpieza y generación de características a partir de los datos.
- `datatables/`: Carpeta que contiene los archivos de datos originales (excluidos del repositorio).

## Funcionalidades

1. **Carga y preprocesamiento de datos**: Limpieza inicial y conversión de formatos.
2. **Generación de variable objetivo**: Identificación de pacientes con retinopatía.
3. **Cálculo de métricas CGM**: Resumen de métricas de glucosa por paciente.
4. **Cálculo de características demográficas y clínicas**: Duración de la diabetes, IMC, entre otros.
5. **Unión de características**: Creación del conjunto de datos final para el modelo.

## Requisitos

- Python 3.8 o superior
- Bibliotecas necesarias:
  - `pandas`
  - `numpy`

## Uso

1. Coloca los archivos de datos en la carpeta `datatables/`.
2. Ejecuta el script `feature-engineering.py`:
   ```bash
   python feature-engineering.py
   ```
3. El conjunto de datos final se guardará como `retinopathy_prediction_dataset.csv`.

## Notas

- Los archivos de datos están excluidos del repositorio mediante `.gitignore`.
- Asegúrate de que los archivos de datos sigan el formato esperado para evitar errores durante la ejecución.

## Licencia

Este proyecto está bajo la licencia MIT.
