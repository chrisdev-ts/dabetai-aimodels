import pandas as pd
import numpy as np


def cargar_y_preprocesar_datos_base(ruta_base):
    """Carga y realiza una limpieza inicial en las tablas de datos base."""
    print("Paso 1: Cargando datos base...")

    archivos = {
        "cgm": f"{ruta_base}IOBP2DeviceCGM.txt",
        "condiciones": f"{ruta_base}IOBP2MedicalCondition.txt",
        "screening": f"{ruta_base}IOBP2DiabScreening.txt",
        "roster": f"{ruta_base}IOBP2PtRoster.txt",
        "hba1c": f"{ruta_base}IOBP2DiabLocalHbA1c.txt",
        "altura_peso": f"{ruta_base}IOBP2HeightWeight.txt",
    }

    try:
        # Cargar todos los archivos en un diccionario de DataFrames
        dfs = {
            nombre: pd.read_csv(ruta, delimiter="|", low_memory=False)
            for nombre, ruta in archivos.items()
        }
    except FileNotFoundError as e:
        print(
            f"Error: No se encontró el archivo {e.filename}. Verifica que la ruta '{ruta_base}' es correcta."
        )
        return None

    # --- Preprocesamiento de fechas ---
    print("Paso 2: Convirtiendo tipos de datos y fechas...")
    dfs["cgm"]["DeviceDtTm"] = pd.to_datetime(
        dfs["cgm"]["DeviceDtTm"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    dfs["condiciones"]["MedCondDiagDt"] = pd.to_datetime(
        dfs["condiciones"]["MedCondDiagDt"], errors="coerce"
    )
    dfs["screening"]["DiagDt"] = pd.to_datetime(
        dfs["screening"]["DiagDt"], errors="coerce"
    )
    dfs["roster"]["EnrollDt"] = pd.to_datetime(
        dfs["roster"]["EnrollDt"], errors="coerce"
    )
    dfs["hba1c"]["HbA1cTestDt"] = pd.to_datetime(
        dfs["hba1c"]["HbA1cTestDt"], errors="coerce"
    )
    dfs["altura_peso"]["WeightAssessDt"] = pd.to_datetime(
        dfs["altura_peso"]["WeightAssessDt"], errors="coerce"
    )

    dfs["cgm"].dropna(subset=["DeviceDtTm"], inplace=True)

    print("Datos base cargados y preprocesados.")
    return dfs


def calcular_metricas_cgm(df_cgm):
    """Calcula métricas de resumen de CGM por paciente."""
    print("Paso 3: Calculando métricas de CGM...")

    def coefficient_of_variation(series):
        return series.std() / series.mean() if series.mean() != 0 else 0

    df_cgm_features = (
        df_cgm.groupby("PtID")["Value"]
        .agg(
            Glucose_Mean="mean",
            Glucose_Std="std",
            Glucose_CV=coefficient_of_variation,
            Time_In_Range_70_180=lambda s: ((s >= 70) & (s <= 180)).mean() * 100,
            Time_Above_180=lambda s: (s > 180).mean() * 100,
            Time_Above_250=lambda s: (s > 250).mean() * 100,
            Time_Below_70=lambda s: (s < 70).mean() * 100,
            Time_Below_54=lambda s: (s < 54).mean() * 100,
        )
        .reset_index()
    )

    print("Métricas de CGM calculadas.")
    return df_cgm_features


def main():
    """Función principal para ejecutar todo el pipeline de preparación de datos."""
    ruta_base_datos = "../data/raw/datatables/"

    dataframes = cargar_y_preprocesar_datos_base(ruta_base_datos)

    if dataframes:
        # --- PASO 1: Crear tabla base de pacientes y variable objetivo ---
        print("Paso 4: Creando variable objetivo (Retinopathy_Status)...")
        df_modelo = dataframes["roster"].copy()

        terminos_retinopatia = ["retinopathy"]
        mascara_retinopatia = dataframes["condiciones"][
            "MedicalCondition"
        ].str.contains("|".join(terminos_retinopatia), case=False, na=False)
        ptids_con_retinopatia = dataframes["condiciones"][mascara_retinopatia][
            "PtID"
        ].unique()

        df_modelo["Retinopathy_Status"] = (
            df_modelo["PtID"].isin(ptids_con_retinopatia).astype(int)
        )

        print("Distribución de la variable objetivo:")
        print(df_modelo["Retinopathy_Status"].value_counts())

        # --- PASO 2: Calcular y unir características ---
        df_cgm_features = calcular_metricas_cgm(dataframes["cgm"])

        # Unir screening para obtener DiagAge y Sex
        df_modelo = pd.merge(
            df_modelo,
            dataframes["screening"][["PtID", "DiagAge", "Sex"]],
            on="PtID",
            how="left",
        )

        # Tomar y unir la última medición de HbA1c
        df_hba1c_final = (
            dataframes["hba1c"]
            .sort_values(by="HbA1cTestDt")
            .drop_duplicates("PtID", keep="last")
        )
        df_modelo = pd.merge(
            df_modelo, df_hba1c_final[["PtID", "HbA1cTestRes"]], on="PtID", how="left"
        )

        # Calcular IMC tomando la última medición de peso/altura
        df_altura_peso = dataframes["altura_peso"].copy()
        df_altura_peso["IMC"] = (df_altura_peso["Weight"] * 0.453592) / (
            (df_altura_peso["Height"] * 0.0254) ** 2
        )
        df_imc_final = df_altura_peso.sort_values(by="WeightAssessDt").drop_duplicates(
            "PtID", keep="last"
        )
        df_modelo = pd.merge(
            df_modelo, df_imc_final[["PtID", "IMC"]], on="PtID", how="left"
        )

        # Unir características de CGM
        df_modelo = pd.merge(df_modelo, df_cgm_features, on="PtID", how="left")

        # --- PASO 3: Limpieza y preparación final ---
        print("Paso 5: Limpiando y finalizando el dataset...")

        # Calcular duración de la diabetes
        df_modelo["Duration_of_Diabetes"] = (
            df_modelo["AgeAsofEnrollDt"] - df_modelo["DiagAge"]
        )

        # Renombrar y seleccionar columnas finales
        df_modelo.rename(columns={"AgeAsofEnrollDt": "Age"}, inplace=True)
        columnas_finales = [
            "PtID",
            "Age",
            "Sex",
            "Duration_of_Diabetes",
            "IMC",
            "HbA1cTestRes",
            "Glucose_Mean",
            "Glucose_Std",
            "Glucose_CV",
            "Time_In_Range_70_180",
            "Time_Above_180",
            "Time_Above_250",
            "Time_Below_70",
            "Time_Below_54",
            "Retinopathy_Status",
        ]
        df_modelo = df_modelo[columnas_finales]

        # Imputar valores faltantes
        for col in ["Duration_of_Diabetes", "IMC", "HbA1cTestRes"]:
            if df_modelo[col].isnull().any():
                median_val = df_modelo[col].median()
                df_modelo[col].fillna(median_val, inplace=True)
                print(
                    f"Valores faltantes en '{col}' imputados con la mediana: {median_val:.2f}"
                )

        # Codificar variables categóricas
        if df_modelo["Sex"].dtype == "object":
            df_modelo["Sex"] = df_modelo["Sex"].astype("category").cat.codes
            print("La columna 'Sex' ha sido codificada a valores numéricos.")

        # --- Resultado final ---
        print("\n--- ¡Dataset final listo para el modelo! ---")
        print("\nInformación del dataset:")
        df_modelo.info()
        print("\nPrimeras 5 filas:")
        print(df_modelo.head())

        nombre_archivo_salida = "../data/processed/retinopathy_model_dataset.csv"
        df_modelo.to_csv(nombre_archivo_salida, index=False)
        print(f"\nDataset final guardado exitosamente en '{nombre_archivo_salida}'")


# Ejecutar el script
if __name__ == "__main__":
    main()
