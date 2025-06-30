import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    make_scorer,
    recall_score,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Ignorar warnings para una salida más limpia
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def run_experiment_suite(dataset_path, target_column, model_configs):
    """
    Carga datos, optimiza, evalúa y analiza múltiples modelos para una complicación.
    """
    print(f"***** INICIANDO SUITE DE EXPERIMENTOS PARA: {target_column} *****")

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{dataset_path}'.")
        return

    X = df.drop(columns=["PtID", target_column])
    y = df[target_column]

    resultados_finales = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    for nombre, config in model_configs.items():
        print(f"\n---------------------------------------------------------")
        print(f"--- Optimizando y evaluando: {nombre} ---")
        print(f"---------------------------------------------------------")

        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring=recall_scorer,
            n_jobs=-1,
        )
        grid_search.fit(X, y)

        print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")

        mejor_modelo = grid_search.best_estimator_
        y_pred = cross_val_predict(mejor_modelo, X, y, cv=cv)

        # --- Reporte detallado para este modelo ---
        complicacion_nombre = target_column.replace("_Status", "")
        class_names = [
            f"Sin {complicacion_nombre} (0)",
            f"Con {complicacion_nombre} (1)",
        ]
        print("\nReporte de clasificación detallado:")
        print(
            classification_report(y, y_pred, target_names=class_names, zero_division=0)
        )

        # --- Análisis de características ---
        mejor_modelo.fit(X, y)
        print(f"\nAnálisis de características para {nombre}:")
        if hasattr(mejor_modelo.named_steps["classifier"], "feature_importances_"):
            importances = mejor_modelo.named_steps["classifier"].feature_importances_
            feature_analysis = pd.Series(importances, index=X.columns).sort_values(
                ascending=False
            )
            print(feature_analysis)
        elif hasattr(mejor_modelo.named_steps["classifier"], "coef_"):
            coeficientes = mejor_modelo.named_steps["classifier"].coef_[0]
            feature_analysis = pd.Series(coeficientes, index=X.columns).sort_values(
                ascending=False
            )
            print(feature_analysis)
        else:
            print("Análisis de características directas no disponible para este modelo")

        # --- Guardar métricas ---
        y_pred_proba = cross_val_predict(
            mejor_modelo, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        reporte_dict = classification_report(
            y, y_pred, output_dict=True, zero_division=0
        )
        resultados_finales[nombre] = {
            "AUC-ROC": roc_auc_score(y, y_pred_proba),
            "Recall (Clase 1)": reporte_dict["1"]["recall"],
            "Precision (Clase 1)": reporte_dict["1"]["precision"],
            "F1-Score (Clase 1)": reporte_dict["1"]["f1-score"],
        }

    # --- Mostrar Tabla Comparativa Final ---
    print("\n\n=========================================================")
    print(f"--- TABLA FINAL: Comparación para {target_column} ---")
    print("=========================================================")
    df_resultados = pd.DataFrame(resultados_finales).T
    df_resultados = df_resultados.sort_values(by="Recall (Clase 1)", ascending=False)
    print(df_resultados)

    if not df_resultados.empty:
        mejor_modelo_nombre = df_resultados.index[0]
        print(f"\n🏆 Modelo ganador general (según Recall): {mejor_modelo_nombre}")


if __name__ == "__main__":
    # Selecciona una de las complicaciones para evaluar
    COMPLICACION_A_EVALUAR = "retinopathy"  # Opciones: "retinopathy", "nephropathy", "neuropathy", "diabetic_foot"

    # --- PANEL DE CONTROL DE MODELOS Y PARÁMETROS ---
    model_configs = {
        "Regresión Logística": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=42)),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=42, max_iter=1000, solver="liblinear"
                        ),
                    ),
                ]
            ),
            "param_grid": {"classifier__C": [0.01, 0.1, 1, 10]},
        },
        "Random Forest": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=42, n_jobs=-1, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [10, 20],
            },
        },
        "LightGBM": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", lgb.LGBMClassifier(random_state=42, verbosity=-1)),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
            },
        },
        "XGBoost": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    (
                        "classifier",
                        xgb.XGBClassifier(
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [5, 10],
            },
        },
        "SVM (RBF Kernel)": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", SVC(probability=True, random_state=42)),
                ]
            ),
            "param_grid": {"classifier__C": [1, 10], "classifier__gamma": ["scale"]},
        },
        "AdaBoost": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", AdaBoostClassifier(random_state=42)),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.1, 1.0],
            },
        },
    }

    target_column_map = {
        "retinopathy": "Retinopathy_Status",
        "nephropathy": "Nephropathy_Status",
        "neuropathy": "Neuropathy_Status",
        "diabetic_foot": "Diabetic_Foot_Status",
    }

    target_column = target_column_map[COMPLICACION_A_EVALUAR]
    dataset_path = f"../data/processed/{COMPLICACION_A_EVALUAR}_model_dataset.csv"

    run_experiment_suite(dataset_path, target_column, model_configs)
