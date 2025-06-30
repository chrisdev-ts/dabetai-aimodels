import pandas as pd
import numpy as np
import warnings

# Ignorar warnings para una salida más limpia
warnings.filterwarnings("ignore", category=FutureWarning)


def cargar_datos_base(ruta_base):
    """Carga y realiza una limpieza inicial en las tablas de datos base."""
    print("--- Cargando y preprocesando datos base ---")
    archivos = {
        "cgm": f"{ruta_base}IOBP2DeviceCGM.txt",
        "condiciones": f"{ruta_base}IOBP2MedicalCondition.txt",
        "screening": f"{ruta_base}IOBP2DiabScreening.txt",
        "roster": f"{ruta_base}IOBP2PtRoster.txt",
        "altura_peso": f"{ruta_base}IOBP2HeightWeight.txt",
        "rand_info": f"{ruta_base}IOBP2RandBaseInfo.txt",
        "socio_econ": f"{ruta_base}IOBP2DiabSocioEcon.txt",
        "fear_hypo": f"{ruta_base}IOBP2PSHFSAdultNoPart2.txt",
        "distress_scale": f"{ruta_base}IOBP2PST1DDS.txt",
    }
    try:
        dfs = {
            nombre: pd.read_csv(ruta, delimiter="|", low_memory=False)
            for nombre, ruta in archivos.items()
        }
    except FileNotFoundError as e:
        print(
            f"Error: No se encontró el archivo {e.filename}. Verifica que la ruta '{ruta_base}' es correcta."
        )
        return None

    # Preprocesamiento de fechas
    dfs["cgm"]["DeviceDtTm"] = pd.to_datetime(
        dfs["cgm"]["DeviceDtTm"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    dfs["cgm"].dropna(subset=["DeviceDtTm"], inplace=True)
    dfs["roster"]["EnrollDt"] = pd.to_datetime(
        dfs["roster"]["EnrollDt"], errors="coerce"
    )
    dfs["screening"]["DiagDt"] = pd.to_datetime(
        dfs["screening"]["DiagDt"], errors="coerce"
    )
    dfs["altura_peso"]["WeightAssessDt"] = pd.to_datetime(
        dfs["altura_peso"]["WeightAssessDt"], errors="coerce"
    )

    print("Datos base cargados exitosamente.")
    return dfs


def calcular_metricas_cgm(df_cgm):
    """Calcula un conjunto estándar de métricas de CGM por paciente."""
    print("Calculando métricas de CGM...")
    if "cgm_features" not in globals():

        def coefficient_of_variation(series):
            return series.std() / series.mean() if series.mean() != 0 else 0

        global cgm_features
        cgm_features = (
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
        print("Métricas de CGM calculadas y cacheadas.")
    else:
        print("Usando métricas de CGM cacheadas.")
    return cgm_features


def preparar_dataset_complicacion(dataframes, config):
    """
    Función genérica para preparar un dataset enriquecido para una complicación específica
    con características accesibles para una app móvil.
    """
    nombre_col_objetivo = config["col_objetivo"]
    terminos_busqueda = config["terminos"]
    nombre_archivo_salida = config["archivo_salida"]

    print(f"\n--- Preparando dataset para: {nombre_col_objetivo} ---")

    # 1. Crear variable objetivo y tabla base
    df_modelo = dataframes["roster"][["PtID", "AgeAsofEnrollDt"]].copy()
    mascara = dataframes["condiciones"]["MedicalCondition"].str.contains(
        "|".join(terminos_busqueda), case=False, na=False
    )
    ptids_con_complicacion = dataframes["condiciones"][mascara]["PtID"].unique()
    df_modelo[nombre_col_objetivo] = (
        df_modelo["PtID"].isin(ptids_con_complicacion).astype(int)
    )

    # 2. Unir características dase accesibles
    df_cgm_features = calcular_metricas_cgm(dataframes["cgm"])
    df_modelo = pd.merge(df_modelo, df_cgm_features, on="PtID", how="left")

    df_screening_subset = dataframes["screening"][
        ["PtID", "DiagAge", "Sex", "InsModPump"]
    ].copy()
    df_screening_subset.rename(columns={"InsModPump": "Is_Pump_User"}, inplace=True)
    df_modelo = pd.merge(df_modelo, df_screening_subset, on="PtID", how="left")

    df_altura_peso = dataframes["altura_peso"].copy()
    df_altura_peso.dropna(subset=["Weight", "Height"], inplace=True)
    df_altura_peso["IMC"] = (df_altura_peso["Weight"] * 0.453592) / (
        (df_altura_peso["Height"] * 0.0254) ** 2
    )
    df_imc_final = df_altura_peso.sort_values(by="WeightAssessDt").drop_duplicates(
        "PtID", keep="last"
    )
    df_modelo = pd.merge(
        df_modelo, df_imc_final[["PtID", "IMC"]], on="PtID", how="left"
    )

    df_modelo.rename(columns={"AgeAsofEnrollDt": "Age"}, inplace=True)
    df_modelo["Duration_of_Diabetes"] = df_modelo["Age"] - df_modelo["DiagAge"]

    # 3. Unir características adicionales accesibles
    ptids_con_hipertension = dataframes["condiciones"][
        dataframes["condiciones"]["MedicalCondition"].str.contains(
            "hypertension", case=False, na=False
        )
    ]["PtID"].unique()
    df_modelo["Has_Hypertension"] = (
        df_modelo["PtID"].isin(ptids_con_hipertension).astype(int)
    )
    df_modelo = pd.merge(
        df_modelo,
        dataframes["rand_info"][["PtID", "TotDlyIns"]],
        on="PtID",
        how="left",
    )

    # 4. AÑADIR LAS NUEVAS CARACTERÍSTICAS
    print("Añadiendo nuevas características de comportamiento y socioeconómicas...")

    # Nivel educativo
    df_socio_econ = dataframes["socio_econ"][["PtID", "RecID", "EducationLevel"]].copy()
    df_socio_econ = df_socio_econ.sort_values("RecID").drop_duplicates(
        "PtID", keep="last"
    )
    education_map = {
        "Less than High School": 0,
        "High school graduate/diploma/GED": 1,
        "Technical/Vocational": 2,
        "Some college but no degree": 2,
        "Associate Degree (AA)": 3,
        "College Graduate (Bachelor's Degree or Equivalent)": 4,
        "Bachelor's Degree (BS,BA,AB)": 4,
        "Advanced Degree (e.g. Masters, PhD, MD)": 5,
        "Master's Degree (MA, MS, MSW, MBA, MPH)": 5,
        "Professional Degree (MD, DDS, DVM, LLB, JD)": 5,
        "Doctorate Degree (PhD, EdD)": 5,
    }
    df_socio_econ["Education_Score"] = df_socio_econ["EducationLevel"].map(
        education_map
    )
    df_modelo = pd.merge(
        df_modelo, df_socio_econ[["PtID", "Education_Score"]], on="PtID", how="left"
    )

    # Miedo a la hipo: Mantener la glucosa alta
    df_fear_hypo = dataframes["fear_hypo"][
        ["PtID", "RecID", "HFSAdultBGAbov150"]
    ].copy()
    df_fear_hypo = df_fear_hypo.sort_values("RecID").drop_duplicates(
        "PtID", keep="last"
    )
    df_fear_hypo.rename(
        columns={"HFSAdultBGAbov150": "Keeps_BG_High_Fear"}, inplace=True
    )
    df_modelo = pd.merge(
        df_modelo, df_fear_hypo.drop(columns="RecID"), on="PtID", how="left"
    )

    # Angustia por diabetes: No comer con cuidado
    df_distress = dataframes["distress_scale"][
        ["PtID", "RecID", "T1DDSNotCarefulEat"]
    ].copy()
    df_distress = df_distress.sort_values("RecID").drop_duplicates("PtID", keep="last")
    df_distress.rename(
        columns={"T1DDSNotCarefulEat": "Not_Careful_Eating_Distress"}, inplace=True
    )
    df_modelo = pd.merge(
        df_modelo, df_distress.drop(columns="RecID"), on="PtID", how="left"
    )

    # 5. Definir la lista final de columnas
    columnas_finales = [
        "PtID",
        "Age",
        "Sex",
        "Duration_of_Diabetes",
        "IMC",
        "Has_Hypertension",
        "TotDlyIns",
        "Is_Pump_User",
        "Glucose_Mean",
        "Glucose_Std",
        "Glucose_CV",
        "Time_In_Range_70_180",
        "Time_Above_180",
        "Time_Above_250",
        "Time_Below_70",
        "Time_Below_54",
        "Education_Score",
        "Keeps_BG_High_Fear",
        "Not_Careful_Eating_Distress",
        nombre_col_objetivo,
    ]

    # 6. Limpieza final
    df_modelo = df_modelo.reindex(columns=columnas_finales)

    for col in df_modelo.columns:
        if pd.api.types.is_numeric_dtype(df_modelo[col]) and col not in [
            "PtID",
            nombre_col_objetivo,
        ]:
            if df_modelo[col].isnull().any():
                median_val = df_modelo[col].median()
                df_modelo[col] = df_modelo[col].fillna(median_val)
    if "Sex" in df_modelo.columns and df_modelo["Sex"].dtype == "object":
        df_modelo["Sex"] = df_modelo["Sex"].astype("category").cat.codes

    # Verificación final del tamaño del dataframe
    if len(df_modelo) > 440:
        print(
            f"ALERTA: El número de filas ({len(df_modelo)}) es mayor al esperado. Revisa las uniones (merges)."
        )
        # Forzar la eliminación de duplicados en el dataframe final como medida de seguridad
        df_modelo.drop_duplicates(subset=["PtID"], keep="last", inplace=True)
        print(f"Número de filas corregido a: {len(df_modelo)}")

    df_modelo.to_csv(nombre_archivo_salida, index=False)
    print(
        f"Dataset para {nombre_col_objetivo} guardado en '{nombre_archivo_salida}'. Filas: {len(df_modelo)}, Columnas: {len(df_modelo.columns)}"
    )


def main():
    """Ejecuta el pipeline completo para generar todos los datasets."""
    ruta_base_datos = "../data/raw/datatables/"

    dataframes = cargar_datos_base(ruta_base_datos)

    if dataframes:
        complicaciones = [
            {
                "terminos": [
                    "retinopathy",
                    "macular edema",
                    "vitreous hemorrhage",
                ],
                "col_objetivo": "Retinopathy_Status",
                "archivo_salida": "../data/processed/retinopathy_model_dataset.csv",
            },
            {
                "terminos": [
                    "nephropathy",
                    "kidney disease",
                    "microalbuminuria",
                    "renal failure",
                    "protein urine present",
                ],
                "col_objetivo": "Nephropathy_Status",
                "archivo_salida": "../data/processed/nephropathy_model_dataset.csv",
            },
            {
                "terminos": ["neuropathy", "polyneuropathy", "gastroparesis"],
                "col_objetivo": "Neuropathy_Status",
                "archivo_salida": "../data/processed/neuropathy_model_dataset.csv",
            },
            {
                "terminos": [
                    "neuropathy",
                    "polyneuropathy",
                    "toe amputation",
                    "foot surgery",
                    "foot pain",
                    "plantar fasciitis",
                    "skin callus",
                    "corns",
                    "foot fracture",
                    "blister of foot",
                    "ingrown toe nail",
                    "hammer toe",
                    "onychomycosis",
                ],
                "col_objetivo": "Diabetic_Foot_Status",
                "archivo_salida": "../data/processed/diabetic_foot_model_dataset.csv",
            },
        ]

        for comp_config in complicaciones:
            preparar_dataset_complicacion(dataframes, comp_config)


if __name__ == "__main__":
    main()
