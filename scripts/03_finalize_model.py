import pandas as pd
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


def get_model_pipeline(model_name, hyperparameters):
    """
    Construye y devuelve el pipeline de un modelo específico con sus hiperparámetros.

    Args:
        model_name (str): El nombre del clasificador ('LogisticRegression' o 'AdaBoost').
        hyperparameters (dict): Un diccionario con los hiperparámetros para el clasificador.

    Returns:
        sklearn.pipeline.Pipeline: El pipeline del modelo listo para ser entrenado.
    """

    # Definimos los pasos comunes del preprocesamiento
    preprocessor_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
    ]

    # Seleccionamos el clasificador basado en el nombre
    if model_name == "LogisticRegression":
        classifier = LogisticRegression(
            random_state=42, max_iter=1000, **hyperparameters
        )
    elif model_name == "AdaBoost":
        classifier = AdaBoostClassifier(random_state=42, **hyperparameters)
    # elif model_name == 'XGBoost':
    #     classifier = xgb.XGBClassifier(random_state=42, **hyperparameters)
    else:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")

    # Combinamos el preprocesador con el clasificador
    # AdaBoost funciona mejor sin escalar los datos, así que ajustamos el pipeline
    if model_name == "AdaBoost":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )
    else:  # Para Regresión Logística y otros, sí escalamos
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )

    return pipeline


def finalize_and_save_model(config):
    """
    Carga datos, entrena el pipeline final especificado en la configuración
    con todos los datos y lo guarda en un archivo .joblib.
    """
    dataset_path = config["dataset_path"]
    model_output_path = config["model_output_path"]
    target_column = config["target_column"]
    best_model_name = config["best_model"]
    hyperparameters = config["hyperparameters"]

    print(f"--- Finalizando modelo para: {target_column} ---")
    print(
        f"Usando el algoritmo: {best_model_name} con los parámetros: {hyperparameters}"
    )

    # Cargar los datos
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["PtID", target_column])
    y = df[target_column]

    # Obtener el pipeline del modelo ganador
    final_pipeline = get_model_pipeline(best_model_name, hyperparameters)

    # Entrenar el pipeline con todos los datos disponibles
    final_pipeline.fit(X, y)
    print("Modelo final entrenado exitosamente.")

    # Guardar el pipeline entrenado en un archivo
    joblib.dump(final_pipeline, model_output_path)
    print(f"--> Modelo guardado en: '{model_output_path}'\n")


# --- PANEL DE CONTROL PRINCIPAL ---
if __name__ == "__main__":

    MODELS_CONFIG = [
        {
            "complication_name": "retinopathy",
            "target_column": "Retinopathy_Status",
            "best_model": "AdaBoost",
            "hyperparameters": {"learning_rate": 0.1, "n_estimators": 50},
        },
        {
            "complication_name": "nephropathy",
            "target_column": "Nephropathy_Status",
            "best_model": "LogisticRegression",
            "hyperparameters": {"C": 0.01},
        },
        {
            "complication_name": "neuropathy",
            "target_column": "Neuropathy_Status",
            "best_model": "LogisticRegression",
            "hyperparameters": {"C": 0.01},
        },
        {
            "complication_name": "diabetic_foot",
            "target_column": "Diabetic_Foot_Status",
            "best_model": "LogisticRegression",
            "hyperparameters": {"C": 0.01},
        },
    ]

    # Bucle para procesar y guardar cada modelo
    for config in MODELS_CONFIG:
        # Añadir las rutas de los archivos a la configuración
        config["dataset_path"] = (
            f"../data/processed/{config['complication_name']}_model_dataset.csv"
        )
        config["model_output_path"] = (
            f"../models/{config['complication_name']}_model.joblib"
        )

        finalize_and_save_model(config)

    print("=============================================")
    print("Todos los modelos han sido finalizados y guardados.")
    print("=============================================")
