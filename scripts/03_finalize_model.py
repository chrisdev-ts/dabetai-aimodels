import pandas as pd
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb


# =============================================================================
# 1. CONFIGURACIONES DE MODELOS Y COMPLICACIONES
#    Centraliza la definición de algoritmos y configuraciones
# =============================================================================
def get_model_configs():
    """
    Devuelve un diccionario con las configuraciones de pipeline para cada modelo.
    Centraliza la definición de los algoritmos disponibles.
    """
    return {
        "Regresión Logística": {
            "classifier": LogisticRegression,
            "default_params": {
                "random_state": 42,
                "max_iter": 1000,
                "solver": "liblinear",
            },
            "requires_scaling": True,
        },
        "Random Forest": {
            "classifier": RandomForestClassifier,
            "default_params": {"random_state": 42, "n_jobs": -1},
            "requires_scaling": False,
        },
        "LightGBM": {
            "classifier": lgb.LGBMClassifier,
            "default_params": {"random_state": 42, "verbosity": -1},
            "requires_scaling": False,
        },
        "XGBoost": {
            "classifier": xgb.XGBClassifier,
            "default_params": {
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            },
            "requires_scaling": False,
        },
        "SVM (RBF Kernel)": {
            "classifier": SVC,
            "default_params": {"probability": True, "random_state": 42},
            "requires_scaling": True,
        },
        "AdaBoost": {
            "classifier": AdaBoostClassifier,
            "default_params": {"random_state": 42},
            "requires_scaling": False,
        },
    }


def get_complicaciones_config():
    """
    Define las configuraciones de complicaciones disponibles.
    Reutiliza la misma estructura que en los otros scripts.
    """
    return {
        "retinopathy": {
            "target_column": "Retinopathy_Status",
            "dataset_path": "../data/processed/retinopathy_model_dataset.csv",
            "model_output_path": "../models/retinopathy_model.joblib",
        },
        "nephropathy": {
            "target_column": "Nephropathy_Status",
            "dataset_path": "../data/processed/nephropathy_model_dataset.csv",
            "model_output_path": "../models/nephropathy_model.joblib",
        },
        "neuropathy": {
            "target_column": "Neuropathy_Status",
            "dataset_path": "../data/processed/neuropathy_model_dataset.csv",
            "model_output_path": "../models/neuropathy_model.joblib",
        },
        "diabetic_foot": {
            "target_column": "Diabetic_Foot_Status",
            "dataset_path": "../data/processed/diabetic_foot_model_dataset.csv",
            "model_output_path": "../models/diabetic_foot_model.joblib",
        },
    }


# =============================================================================
# 2. CONSTRUCCIÓN DE PIPELINES
#    Funciones para construir y configurar los pipelines de ML
# =============================================================================
def get_model_pipeline(model_name, hyperparameters):
    """
    Construye y devuelve el pipeline de un modelo específico con sus hiperparámetros.
    Ahora soporta todos los algoritmos disponibles y decide automáticamente si necesita escalado.

    Args:
        model_name (str): El nombre del clasificador según get_model_configs().
        hyperparameters (dict): Un diccionario con los hiperparámetros para el clasificador.

    Returns:
        sklearn.pipeline.Pipeline: El pipeline del modelo listo para ser entrenado.
    """
    model_configs = get_model_configs()

    if model_name not in model_configs:
        available_models = list(model_configs.keys())
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. Modelos disponibles: {available_models}"
        )

    config = model_configs[model_name]

    # Combinar parámetros por defecto con hiperparámetros específicos
    final_params = {**config["default_params"], **hyperparameters}

    # Crear el clasificador
    classifier = config["classifier"](**final_params)

    # Construir el pipeline según si requiere escalado o no
    if config["requires_scaling"]:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )

    return pipeline


# =============================================================================
# 3. FUNCIONES DE FINALIZACIÓN
#    Funciones para entrenar y guardar los modelos finales
# =============================================================================
def finalize_and_save_model(complication_name, model_name, hyperparameters):
    """
    Carga datos, entrena el pipeline final especificado y lo guarda en un archivo .joblib.

    Args:
        complication_name (str): Nombre de la complicación ('retinopathy', 'nephropathy', etc.)
        model_name (str): Nombre del algoritmo según get_model_configs()
        hyperparameters (dict): Hiperparámetros específicos para el modelo
    """
    complicaciones_config = get_complicaciones_config()

    if complication_name not in complicaciones_config:
        available_complications = list(complicaciones_config.keys())
        raise ValueError(
            f"Complicación '{complication_name}' no reconocida. Disponibles: {available_complications}"
        )

    config = complicaciones_config[complication_name]
    dataset_path = config["dataset_path"]
    model_output_path = config["model_output_path"]
    target_column = config["target_column"]

    print(f"Finalizando modelo para {complication_name.upper()}")

    # 1. Cargar los datos
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{dataset_path}'")
        return False

    X = df.drop(columns=["PtID", target_column])
    y = df[target_column]

    print(f"   Dataset: {len(df)} filas, {len(X.columns)} características")
    print(f"   Algoritmo: {model_name}")

    # 2. Obtener el pipeline del modelo ganador
    try:
        final_pipeline = get_model_pipeline(model_name, hyperparameters)
    except Exception as e:
        print(f"Error al construir el pipeline: {e}")
        return False

    # 3. Entrenar el pipeline con todos los datos disponibles
    print("   Entrenando modelo final...")
    try:
        final_pipeline.fit(X, y)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return False

    # 4. Guardar el pipeline entrenado en un archivo
    try:
        joblib.dump(final_pipeline, model_output_path)
        print(f"   Modelo guardado exitosamente")
        return True
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
        return False


def finalize_multiple_models(models_config):
    """
    Finaliza múltiples modelos según una lista de configuraciones.

    Args:
        models_config (list): Lista de diccionarios con configuraciones de modelos
    """
    print(f"\nFinalizando {len(models_config)} modelos\n")

    successful = 0
    failed = 0

    for i, config in enumerate(models_config, 1):
        result = finalize_and_save_model(
            complication_name=config["complication_name"],
            model_name=config["model_name"],
            hyperparameters=config["hyperparameters"],
        )

        if result:
            successful += 1
        else:
            failed += 1

        if i < len(models_config):  # Solo agregar línea si no es el último
            print()

    print(f"\nResumen: {successful} exitosos, {failed} fallidos")

    if failed == 0:
        print("Todos los modelos finalizados exitosamente")
    else:
        print("Algunos modelos fallaron. Revisa los mensajes anteriores.")


def finalize_single_model(complication_name, model_name, hyperparameters):
    """
    Finaliza un solo modelo. Útil para pruebas rápidas o cuando solo necesitas un modelo.

    Args:
        complication_name (str): Nombre de la complicación
        model_name (str): Nombre del algoritmo
        hyperparameters (dict): Hiperparámetros del modelo
    """
    print(f"Finalizando modelo individual\n")

    result = finalize_and_save_model(complication_name, model_name, hyperparameters)

    if result:
        print(f"\nModelo finalizado exitosamente")
    else:
        print(f"\nError al finalizar el modelo")

    return result


# =============================================================================
# 4. FUNCIONES DE UTILIDAD Y VISUALIZACIÓN
#    Funciones auxiliares para mostrar opciones y compatibilidad
# =============================================================================
def show_available_options():
    """
    Muestra las opciones disponibles de complicaciones y algoritmos.
    """
    print(f"\nOpciones disponibles\n")

    print("Complicaciones:")
    complicaciones = get_complicaciones_config()
    for name, config in complicaciones.items():
        print(f"   • {name}")

    print("\nAlgoritmos:")
    modelos = get_model_configs()
    for name, config in modelos.items():
        print(f"   • {name}")

    print(f"\n{'-'*40}")


# Mantener compatibilidad con la versión anterior
def finalize_and_save_model_legacy(config):
    """
    Función legacy para mantener compatibilidad con configuraciones anteriores.
    """
    return finalize_and_save_model(
        complication_name=config.get("complication_name"),
        model_name=config.get("best_model"),
        hyperparameters=config.get("hyperparameters", {}),
    )


# =============================================================================
# PANEL DE CONTROL PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    # Configuración de modelos ganadores (basado en experimentos previos)
    MODELS_CONFIG = [
        {
            "complication_name": "retinopathy",
            "model_name": "AdaBoost",
            "hyperparameters": {"learning_rate": 0.1, "n_estimators": 50},
        },
        {
            "complication_name": "nephropathy",
            "model_name": "Regresión Logística",
            "hyperparameters": {"C": 0.01},
        },
        {
            "complication_name": "neuropathy",
            "model_name": "Regresión Logística",
            "hyperparameters": {"C": 0.01},
        },
        {
            "complication_name": "diabetic_foot",
            "model_name": "Regresión Logística",
            "hyperparameters": {"C": 0.01},
        },
    ]

    # Opciones de ejecución:

    # Opción 1: Mostrar opciones disponibles
    # show_available_options()

    # Opción 2: Finalizar todos los modelos configurados
    finalize_multiple_models(MODELS_CONFIG)

    # Opción 3: Finalizar un solo modelo (comentado por defecto)
    # finalize_single_model(
    #     complication_name="retinopathy",
    #     model_name="AdaBoost",
    #     hyperparameters={"learning_rate": 0.1, "n_estimators": 50}
    # )
