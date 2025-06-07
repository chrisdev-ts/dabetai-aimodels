import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib


def finalizar_y_guardar_modelo(dataset_path, model_output_path):
    """
    Entrena el pipeline final (Regresión Logística) con todos los datos y lo guarda.
    """
    print("--- Entrenando el modelo final con todos los datos ---")

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["PtID", "Retinopathy_Status"])
    y = df["Retinopathy_Status"]

    # Define el pipeline del modelo
    pipeline_final = Pipeline(
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
    )

    # Entrenar el pipeline con todos los datos disponibles
    pipeline_final.fit(X, y)

    print("Modelo final entrenado exitosamente.")

    # Guardar el pipeline entrenado en un archivo
    joblib.dump(pipeline_final, model_output_path)

    print(
        f"\nEl modelo de Regresión Logística ha sido guardado en: '{model_output_path}'"
    )


# --- Ejecutar el Script Principal ---
if __name__ == "__main__":
    nombre_archivo_dataset = "../data/processed/retinopathy_model_dataset.csv"
    nombre_archivo_modelo = "../models/rl_prediction_model.joblib"
    finalizar_y_guardar_modelo(nombre_archivo_dataset, nombre_archivo_modelo)
