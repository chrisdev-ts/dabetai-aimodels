import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
    confusion_matrix,
    roc_curve,
    auc,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)


# =============================================================================
# 1. FÁBRICA DE MODELOS
#    Centraliza la definición de los algoritmos. Para añadir un nuevo
#    algoritmo, solo necesitas agregarlo aquí.
# =============================================================================
def get_model_configs():
    """
    Devuelve un diccionario con las configuraciones de pipeline y parámetros
    para cada modelo que se desea probar.
    """
    return {
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
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1, 10],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
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
            "param_grid": {
                "classifier__C": [1, 10],
                "classifier__gamma": ["scale"],
            },
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


# =============================================================================
# 2. FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN
# =============================================================================
def show_feature_importance(model, feature_names, model_name, phase=""):
    """Extrae y muestra la importancia de las características del modelo."""
    print(f"\n--- Importancia de características para {model_name} {phase} ---")

    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        print(f"No se puede extraer importancia para {model_name}")
        return

    feature_importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print(feature_importance_df.to_string())

    return feature_importance_df


def show_smote_info(model, X_original, y_original, model_name):
    """Muestra información detallada sobre el balanceamiento aplicado por SMOTE."""
    print(f"\n--- Información de balanceamiento SMOTE para {model_name} ---")

    # Distribución original
    original_distribution = pd.Series(y_original).value_counts().sort_index()
    print(f"Distribución original de clases:")
    for clase, count in original_distribution.items():
        percentage = (count / len(y_original)) * 100
        print(f"  Clase {clase}: {count} muestras ({percentage:.1f}%)")

    # Obtener parámetros de SMOTE del mejor modelo
    if hasattr(model, "named_steps") and "smote" in model.named_steps:
        smote_step = model.named_steps["smote"]
        sampling_strategy = smote_step.sampling_strategy

        print(f"\nEstrategia de muestreo SMOTE: {sampling_strategy}")

        # Simular aplicación de SMOTE para mostrar distribución final
        try:
            from collections import Counter

            # Aplicar solo imputer si existe
            X_processed = X_original.copy()
            if "imputer" in model.named_steps:
                X_processed = model.named_steps["imputer"].transform(X_processed)

            # Aplicar SMOTE
            X_resampled, y_resampled = smote_step.fit_resample(X_processed, y_original)

            # Distribución después de SMOTE
            final_distribution = pd.Series(y_resampled).value_counts().sort_index()
            print(f"\nDistribución después de SMOTE:")
            for clase, count in final_distribution.items():
                percentage = (count / len(y_resampled)) * 100
                print(f"  Clase {clase}: {count} muestras ({percentage:.1f}%)")

            print(f"\nCambio en el tamaño del dataset:")
            print(f"  Original: {len(y_original)} muestras")
            print(f"  Después de SMOTE: {len(y_resampled)} muestras")
            print(
                f"  Incremento: +{len(y_resampled) - len(y_original)} muestras ({((len(y_resampled) / len(y_original)) - 1) * 100:.1f}%)"
            )

        except Exception as e:
            print(f"  No se pudo simular SMOTE: {e}")
    else:
        print("  No se encontró configuración de SMOTE en el modelo")


def generate_final_reports(
    y_true,
    y_pred,
    y_proba,
    class_names,
    target_column,
    model_name,
    model=None,
    feature_names=None,
):
    """Genera y guarda reportes completos: matriz de confusión, curva ROC, métricas, reporte de clasificación e importancia de características."""
    output_dir = f'../reports/figures/{target_column.replace("_Status", "")}'
    os.makedirs(output_dir, exist_ok=True)

    # 1. MATRIZ DE CONFUSIÓN
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.title(
        f"Matriz de confusión final - {model_name}", fontsize=18, fontweight="bold"
    )
    plt.ylabel("Valor real", fontsize=14)
    plt.xlabel("Predicción", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"FINAL_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. CURVA ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=3, label=f"{model_name} (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Línea de azar")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de falsos positivos (1 - especificidad)", fontsize=14)
    plt.ylabel("Tasa de verdaderos positivos (sensibilidad)", fontsize=14)
    plt.title(f"Curva ROC - {model_name}", fontsize=18, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"FINAL_roc_curve.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. MÉTRICAS DETALLADAS
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        balanced_accuracy_score,
    )

    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

    # Crear gráfico de métricas
    metrics_names = [
        "Accuracy",
        "Balanced\naccuracy",
        "Precision\n(clase 1)",
        "Recall/sensitivity\n(clase 1)",
        "F1-score\n(clase 1)",
        "Specificity\n(clase 0)",
        "AUC-ROC",
    ]
    metrics_values = [
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        f1,
        specificity,
        roc_auc,
    ]

    plt.figure(figsize=(12, 8))
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    bars = plt.bar(
        metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor="black"
    )

    # Añadir valores sobre las barras
    for bar, value in zip(bars, metrics_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.ylim(0, 1.1)
    plt.title(f"Métricas de rendimiento - {model_name}", fontsize=18, fontweight="bold")
    plt.ylabel("Valor de la métrica", fontsize=14)
    plt.xlabel("Métricas", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"FINAL_metrics_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. REPORTE DE CLASIFICACIÓN DETALLADO
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Crear DataFrame del reporte
    report_df = pd.DataFrame(class_report).transpose()

    # Crear tabla visual del reporte
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    # Preparar datos para la tabla
    table_data = []
    for index, row in report_df.iterrows():
        if index in ["accuracy", "macro avg", "weighted avg"] or index in class_names:
            if index == "accuracy":
                table_data.append(
                    [
                        index.title(),
                        f"{row['precision']:.3f}",
                        "",
                        "",
                        f"{row['support']:.0f}",
                    ]
                )
            else:
                table_data.append(
                    [
                        index.title(),
                        f"{row['precision']:.3f}",
                        f"{row['recall']:.3f}",
                        f"{row['f1-score']:.3f}",
                        f"{row['support']:.0f}",
                    ]
                )

    # Crear tabla
    table = ax.table(
        cellText=table_data,
        colLabels=["Clase", "Precision", "Recall", "F1-score", "Support"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Estilizar tabla
    for i in range(len(table_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

    plt.title(
        f"Reporte de clasificación detallado - {model_name}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"FINAL_classification_report.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 5. GRÁFICA DE IMPORTANCIA DE CARACTERÍSTICAS
    if model is not None and feature_names is not None:
        classifier = model.named_steps["classifier"]

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = None

        if importances is not None:
            # Crear DataFrame con importancias
            feature_importance_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

            # Mostrar TODAS las características disponibles
            all_features = feature_importance_df

            plt.figure(figsize=(12, max(10, len(all_features) * 0.5)))
            colors = plt.cm.viridis(np.linspace(0, 1, len(all_features)))
            bars = plt.barh(
                range(len(all_features)),
                all_features["Importance"],
                color=colors,
                alpha=0.8,
                edgecolor="black",
            )

            # Configurar etiquetas
            plt.yticks(range(len(all_features)), all_features["Feature"])
            plt.xlabel("Importancia de la característica", fontsize=14)
            plt.ylabel("Características", fontsize=14)
            plt.title(
                f"Importancia de todas las características ({len(all_features)}) - {model_name}",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            # Añadir valores en las barras
            for i, (bar, value) in enumerate(zip(bars, all_features["Importance"])):
                plt.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=max(
                        8, min(10, 120 // len(all_features))
                    ),  # Ajustar tamaño de fuente según número de características
                    fontweight="bold",
                )

            # Invertir el orden del eje y para que la característica más importante esté arriba
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"FINAL_feature_importance.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # 6. Imprimir resumen en consola
    print(f"\n📊 Resumen de métricas finales para {model_name}:")
    print(f"{'='*60}")
    print(f"Accuracy general:           {accuracy:.3f}")
    print(f"Balanced accuracy:          {balanced_accuracy:.3f}")
    print(f"Precision (clase 1):        {precision:.3f}")
    print(f"Recall/sensibilidad:        {recall:.3f}")
    print(f"F1-score (clase 1):         {f1:.3f}")
    print(f"Especificidad (clase 0):    {specificity:.3f}")
    print(f"AUC-ROC:                   {roc_auc:.3f}")
    print(f"{'='*60}")

    # Interpretación clínica
    print(f"\n🏥 Interpretación clínica:")
    print(
        f"• Sensibilidad (recall): {recall:.1%} de los casos positivos son detectados correctamente"
    )
    print(
        f"• Especificidad: {specificity:.1%} de los casos negativos son identificados correctamente"
    )
    print(f"• Precisión: {precision:.1%} de las predicciones positivas son correctas")

    # Mensaje sobre archivos generados
    print(f"\n🎯 Reportes detallados del modelo ganador generados en: {output_dir}")
    print("   🏆 FINAL_confusion_matrix.png - Matriz de confusión")
    print("   🏆 FINAL_roc_curve.png - Curva ROC")
    print("   🏆 FINAL_metrics_summary.png - Resumen de métricas")
    print("   🏆 FINAL_classification_report.png - Reporte de clasificación")
    print("   🏆 FINAL_feature_importance.png - Importancia de características")


def generate_comparative_reports(
    df_resultados, target_column, best_models, X_test, y_test
):
    """Genera reportes comparativos cuando se ejecutan múltiples algoritmos."""
    output_dir = f'../reports/figures/{target_column.replace("_Status", "")}'
    os.makedirs(output_dir, exist_ok=True)

    if len(df_resultados) < 2:
        return  # No generar reportes comparativos si solo hay un modelo

    # 1. GRÁFICO COMPARATIVO DE MÉTRICAS DE VALIDACIÓN CRUZADA
    cv_metrics = [
        "Recall (CV en Train)",
        "Precision (CV en Train)",
        "F1-Score (CV en Train)",
        "AUC-ROC (CV en Train)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, metric in enumerate(cv_metrics):
        ax = axes[i]

        # Ordenar por la métrica actual
        sorted_data = df_resultados.sort_values(by=metric, ascending=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_data)))
        bars = ax.barh(
            range(len(sorted_data)),
            sorted_data[metric],
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )

        # Configurar etiquetas
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data.index, fontsize=10)
        ax.set_xlabel(metric.replace("(CV en Train)", ""), fontsize=12)
        ax.set_title(f"{metric}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Añadir valores en las barras
        for j, (bar, value) in enumerate(zip(bars, sorted_data[metric])):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle(
        f"Comparación de métricas de validación cruzada - {target_column}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"COMPARATIVE_cv_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. GRÁFICO COMPARATIVO DE MÉTRICAS DE TEST FINAL
    test_metrics = [
        "Recall (Test Final)",
        "Precision (Test Final)",
        "F1-Score (Test Final)",
        "AUC-ROC (Test Final)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, metric in enumerate(test_metrics):
        ax = axes[i]

        # Ordenar por la métrica actual
        sorted_data = df_resultados.sort_values(by=metric, ascending=True)

        colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_data)))
        bars = ax.barh(
            range(len(sorted_data)),
            sorted_data[metric],
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )

        # Configurar etiquetas
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data.index, fontsize=10)
        ax.set_xlabel(metric.replace("(Test Final)", ""), fontsize=12)
        ax.set_title(f"{metric}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Añadir valores en las barras
        for j, (bar, value) in enumerate(zip(bars, sorted_data[metric])):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle(
        f"Comparación de métricas de test final - {target_column}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"COMPARATIVE_test_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. GRÁFICO COMPARATIVO DE CURVAS ROC
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(best_models)))

    for i, (model_name, model) in enumerate(best_models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, color=colors[i], lw=3, label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Línea de azar")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de falsos positivos (1 - especificidad)", fontsize=14)
    plt.ylabel("Tasa de verdaderos positivos (sensibilidad)", fontsize=14)
    plt.title(
        f"Comparación de curvas ROC - {target_column}", fontsize=18, fontweight="bold"
    )
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"COMPARATIVE_roc_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. TABLA COMPARATIVA DETALLADA
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("tight")
    ax.axis("off")

    # Preparar datos para la tabla
    table_data = []
    for model_name in df_resultados.index:
        row_data = [model_name]
        for col in df_resultados.columns:
            row_data.append(f"{df_resultados.loc[model_name, col]:.3f}")
        table_data.append(row_data)

    # Crear tabla
    columns = ["Modelo"] + list(df_resultados.columns)
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Estilizar tabla
    for i in range(len(table_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor("#2E86AB")
                cell.set_text_props(weight="bold", color="white")
            elif j == 0:  # Primera columna (nombres de modelos)
                cell.set_facecolor("#A23B72")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#F18F01" if i % 2 == 0 else "#C73E1D")
                cell.set_text_props(color="white")

    plt.title(
        f"Tabla comparativa completa - {target_column}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"COMPARATIVE_table.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\n📊 Reportes comparativos generados en: {output_dir}")
    print("   📈 COMPARATIVE_cv_metrics.png - Comparación métricas validación cruzada")
    print("   📈 COMPARATIVE_test_metrics.png - Comparación métricas test final")
    print("   📈 COMPARATIVE_roc_curves.png - Comparación curvas ROC")
    print("   📈 COMPARATIVE_table.png - Tabla comparativa completa")


# =============================================================================
# 3. MOTOR PRINCIPAL DEL EXPERIMENTO
# =============================================================================
def run_full_evaluation(complication_name, target_column, algorithms_to_run):
    """
    Ejecuta el pipeline completo, reportando tanto el rendimiento de la
    validación cruzada como el de la prueba final.
    """
    print(f"\n{'='*80}")
    print(
        f"***** INICIANDO EVALUACIÓN COMPLETA PARA: {complication_name.upper()} *****"
    )
    print(f"{'='*80}")

    # Cargar datos
    dataset_path = f"../data/processed/{complication_name}_model_dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset '{dataset_path}' cargado exitosamente.")
    except FileNotFoundError:
        print(
            f"Error: No se encontró el archivo '{dataset_path}'. Saltando esta complicación."
        )
        return

    X = df.drop(columns=["PtID", target_column])
    y = df[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"\nDatos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba."
    )

    all_model_configs = get_model_configs()
    models_to_run = {
        name: config
        for name, config in all_model_configs.items()
        if name in algorithms_to_run
    }

    resultados = []
    best_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    print(
        f"\n--- FASE 1: Optimización con Validación Cruzada (sobre {len(X_train)} registros de entrenamiento) ---"
    )
    for nombre, config in models_to_run.items():
        print(f"  -> Optimizando: {nombre}...")

        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring=recall_scorer,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_models[nombre] = grid_search.best_estimator_

        print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")

        # Mostrar información de SMOTE
        show_smote_info(grid_search.best_estimator_, X_train, y_train, nombre)

        # Mostrar importancia de características después de la optimización
        show_feature_importance(
            grid_search.best_estimator_,
            X_train.columns,
            nombre,
            "(optimización - validación cruzada)",
        )

        # Obtener métricas adicionales de validación cruzada
        # Realizar predicciones con validación cruzada para obtener todas las métricas
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import precision_score, f1_score

        y_pred_cv = cross_val_predict(
            grid_search.best_estimator_, X_train, y_train, cv=cv
        )
        y_proba_cv = cross_val_predict(
            grid_search.best_estimator_, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]

        precision_cv = precision_score(y_train, y_pred_cv, zero_division=0)
        f1_cv = f1_score(y_train, y_pred_cv, zero_division=0)
        auc_cv = roc_auc_score(y_train, y_proba_cv)

        resultados.append(
            {
                "Modelo": nombre,
                "Recall (CV en Train)": grid_search.best_score_,
                "Precision (CV en Train)": precision_cv,
                "F1-Score (CV en Train)": f1_cv,
                "AUC-ROC (CV en Train)": auc_cv,
            }
        )

    # Crear un DataFrame inicial con los resultados de la CV
    df_resultados = pd.DataFrame(resultados).set_index("Modelo")

    print(
        f"\n--- FASE 2: Evaluación Final (sobre {len(X_test)} registros de prueba) ---"
    )
    test_results = {}
    for nombre, model in best_models.items():
        print(f"  -> Evaluando: {nombre}...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        reporte = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        metrics = reporte.get("1", {"recall": 0.0, "precision": 0.0, "f1-score": 0.0})

        test_results[nombre] = {
            "Recall (Test Final)": metrics["recall"],
            "Precision (Test Final)": metrics["precision"],
            "F1-Score (Test Final)": metrics["f1-score"],
            "AUC-ROC (Test Final)": roc_auc_score(y_test, y_proba),
        }

    # Unir los resultados del test al DataFrame principal
    df_resultados = df_resultados.join(pd.DataFrame(test_results).T)
    df_resultados = df_resultados.sort_values(by="Recall (Test Final)", ascending=False)

    print(f"\n--- TABLA COMPARATIVA FINAL para {target_column} ---")
    print(df_resultados)

    # Generar reportes comparativos generales si hay múltiples modelos
    if len(df_resultados) > 1:
        print(f"\n--- Generando reportes comparativos entre todos los algoritmos ---")
        generate_comparative_reports(
            df_resultados, target_column, best_models, X_test, y_test
        )

    if not df_resultados.empty:
        mejor_modelo_nombre = df_resultados.index[0]
        print(f"\n🏆 Modelo ganador (según Recall en Test): {mejor_modelo_nombre}")

        # Análisis de características y reportes visuales para el modelo ganador
        modelo_ganador = best_models[mejor_modelo_nombre]
        show_feature_importance(
            modelo_ganador,
            X_train.columns,
            mejor_modelo_nombre,
            "(evaluación final - test set)",
        )

        print("\n--- Generando reportes detallados para el modelo ganador ---")
        y_pred_ganador = modelo_ganador.predict(X_test)
        y_proba_ganador = modelo_ganador.predict_proba(X_test)[:, 1]
        class_names = [f"Sin {complication_name}", f"Con {complication_name}"]
        generate_final_reports(
            y_test,
            y_pred_ganador,
            y_proba_ganador,
            class_names,
            target_column,
            mejor_modelo_nombre,
            modelo_ganador,
            X_train.columns,
        )

    print(f"\n{'='*80}")
    print(f"✅ EVALUACIÓN COMPLETA FINALIZADA PARA {complication_name.upper()}")
    print(f"{'='*80}")


# =============================================================================
# 4. PANEL DE CONTROL PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    COMPLICATION_TO_RUN = "retinopathy"
    ALGORITHMS_TO_RUN = [
        "AdaBoost",
        "Regresión Logística",
        "Random Forest",
        "LightGBM",
        "XGBoost",
        "SVM (RBF Kernel)",
    ]

    target_column_map = {
        "retinopathy": "Retinopathy_Status",
        "nephropathy": "Nephropathy_Status",
        "neuropathy": "Neuropathy_Status",
        "diabetic_foot": "Diabetic_Foot_Status",
    }

    if COMPLICATION_TO_RUN in target_column_map:
        run_full_evaluation(
            complication_name=COMPLICATION_TO_RUN,
            target_column=target_column_map[COMPLICATION_TO_RUN],
            algorithms_to_run=ALGORITHMS_TO_RUN,
        )
    else:
        print(
            f"Error: El nombre de la complicación '{COMPLICATION_TO_RUN}' no es válido."
        )
