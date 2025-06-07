import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def comparar_modelos(dataset_path):
    """
    Carga los datos, entrena y evalúa múltiples modelos.
    """
    # --- PASO 1: Cargar el dataset limpio ---
    print("--- PASO 1: Cargando el dataset preparado para el modelo ---")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{dataset_path}'.")
        return

    # --- PASO 2: Separar características (X) y variable objetivo (y) ---
    X = df.drop(columns=["PtID", "Retinopathy_Status"])
    y = df["Retinopathy_Status"]

    # --- PASO 3: Definir los pipelines de los modelos a comparar ---
    print("\n--- PASO 3: Configurando los pipelines de los modelos ---")

    modelos = {
        "Regresión Logística": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=42, max_iter=1000, class_weight="balanced"
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("smote", SMOTE(random_state=42)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("smote", SMOTE(random_state=42)),
                ("classifier", lgb.LGBMClassifier(random_state=42)),
            ]
        ),
    }

    # --- PASO 4: Ejecutar la validación cruzada para cada modelo ---
    resultados = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for nombre, pipeline in modelos.items():
        print(f"\n--- Entrenando y evaluando: {nombre} ---")

        # Usamos cross_val_predict para obtener predicciones y probabilidades para cada pliegue
        from sklearn.model_selection import cross_val_predict

        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
        y_pred_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[
            :, 1
        ]

        # Calcular métricas
        auc = roc_auc_score(y, y_pred_proba)
        reporte = classification_report(
            y, y_pred, target_names=["Sin RD (0)", "Con RD (1)"], output_dict=True
        )

        # Guardar las métricas que nos interesan
        resultados[nombre] = {
            "AUC-ROC": auc,
            "Recall (Clase 1)": reporte["Con RD (1)"]["recall"],
            "Precision (Clase 1)": reporte["Con RD (1)"]["precision"],
            "F1-Score (Clase 1)": reporte["Con RD (1)"]["f1-score"],
            "Accuracy": reporte["accuracy"],
        }

        print(f"Resultados para {nombre} calculados.")
        print(f"  -> AUC-ROC: {auc:.4f}")
        print(f"  -> Recall (Con RD): {reporte['Con RD (1)']['recall']:.4f}")

    # --- PASO 5: Mostrar la tabla comparativa final ---
    print("\n\n--- PASO 5: Tabla comparativa de rendimiento de modelos ---")

    df_resultados = pd.DataFrame(
        resultados
    ).T  # .T transpone la tabla para mejor visualización
    df_resultados = df_resultados.sort_values(by="AUC-ROC", ascending=False)

    print(df_resultados)

    mejor_modelo_nombre = df_resultados.index[0]
    print(f"\nEl mejor modelo según el AUC-ROC es: {mejor_modelo_nombre}")

    # --- (Opcional) PASO 6: Analizar Características del Mejor Modelo ---
    print(
        f"\n--- PASO 6: Analizando características del mejor modelo ({mejor_modelo_nombre}) ---"
    )

    mejor_pipeline = modelos[mejor_modelo_nombre]
    mejor_pipeline.fit(X, y)

    # Comprobar si el modelo tiene 'feature_importances_' o 'coef_'
    if hasattr(mejor_pipeline.named_steps["classifier"], "feature_importances_"):
        importances = mejor_pipeline.named_steps["classifier"].feature_importances_
        feature_analysis = pd.Series(importances, index=X.columns).sort_values(
            ascending=False
        )
        print("\nImportancia de cada característica:")
    elif hasattr(mejor_pipeline.named_steps["classifier"], "coef_"):
        coeficientes = mejor_pipeline.named_steps["classifier"].coef_[0]
        feature_analysis = pd.Series(coeficientes, index=X.columns).sort_values(
            ascending=False
        )
        print("\nCoeficientes de cada característica:")
    else:
        feature_analysis = (
            "Análisis de características no disponible para este tipo de modelo."
        )

    print(feature_analysis)


# --- Ejecutar el Script Principal ---
if __name__ == "__main__":
    nombre_archivo_dataset = "../data/processed/retinopathy_model_dataset.csv"
    comparar_modelos(nombre_archivo_dataset)
