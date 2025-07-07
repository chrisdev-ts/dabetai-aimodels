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


def get_feature_configs():
    """
    Define las configuraciones de características para cada tipo de complicación.
    Esto permite personalizar qué características incluir según la relevancia clínica.
    """
    return {
        "retinopathy": {
            "base_features": [
                "PtID",
                "Age",
                "Sex",
                "Duration_of_Diabetes",
            ],
            "metabolic_features": ["IMC", "Has_Hypertension"],
            "glucose_features": [
                "Glucose_Mean",
                "Glucose_Std",
                "Glucose_CV",
                "Time_In_Range_70_180",
                "Time_Above_180",
                "Time_Above_250",
                "Time_Below_70",
                "Time_Below_54",
            ],
            "behavioral_features": [],
            "insulin_features": [],
        },
        "nephropathy": {
            "base_features": [
                "PtID",
                "Age",
                "Sex",
                "Duration_of_Diabetes",
            ],
            "metabolic_features": ["IMC", "Has_Hypertension"],
            "glucose_features": [
                "Glucose_Mean",
                "Glucose_Std",
                "Glucose_CV",
                "Time_In_Range_70_180",
                "Time_Above_180",
                "Time_Above_250",
                "Time_Below_70",
                "Time_Below_54",
            ],
            "behavioral_features": [
                "Education_Score",
                "Keeps_BG_High_Fear",
                "Not_Careful_Eating_Distress",
            ],
            "insulin_features": ["TotDlyIns", "Is_Pump_User"],
        },
        "neuropathy": {
            "base_features": [
                "PtID",
                "Age",
                "Sex",
                "Duration_of_Diabetes",
            ],
            "metabolic_features": ["IMC", "Has_Hypertension"],
            "glucose_features": [
                "Glucose_Mean",
                "Glucose_Std",
                "Glucose_CV",
                "Time_In_Range_70_180",
                "Time_Above_180",
                "Time_Above_250",
                "Time_Below_70",
                "Time_Below_54",
            ],
            "behavioral_features": [
                "Education_Score",
                "Keeps_BG_High_Fear",
                "Not_Careful_Eating_Distress",
            ],
            "insulin_features": ["TotDlyIns", "Is_Pump_User"],
        },
        "diabetic_foot": {
            "base_features": [
                "PtID",
                "Age",
                "Sex",
                "Duration_of_Diabetes",
            ],
            "metabolic_features": ["IMC", "Has_Hypertension"],
            "glucose_features": [
                "Glucose_Mean",
                "Glucose_Std",
                "Glucose_CV",
                "Time_In_Range_70_180",
                "Time_Above_180",
                "Time_Above_250",
                "Time_Below_70",
                "Time_Below_54",
            ],
            "behavioral_features": [
                "Education_Score",
                "Keeps_BG_High_Fear",
                "Not_Careful_Eating_Distress",
            ],
            "insulin_features": ["TotDlyIns", "Is_Pump_User"],
        },
    }


def get_complicaciones_config():
    """
    Define las configuraciones de complicaciones para evitar duplicación de código.
    Retorna la lista de configuraciones para todas las complicaciones.
    """
    return [
        {
            "tipo": "retinopathy",
            "terminos": [
                "retinopathy",
                "macular edema",
                "vitreous hemorrhage",
            ],
            "col_objetivo": "Retinopathy_Status",
            "archivo_salida": "../data/processed/retinopathy_model_dataset.csv",
        },
        {
            "tipo": "nephropathy",
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
            "tipo": "neuropathy",
            "terminos": ["neuropathy", "polyneuropathy", "gastroparesis"],
            "col_objetivo": "Neuropathy_Status",
            "archivo_salida": "../data/processed/neuropathy_model_dataset.csv",
        },
        {
            "tipo": "diabetic_foot",
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


def calcular_caracteristicas_adicionales(df_modelo, dataframes):
    """Calcula características adicionales que pueden ser relevantes para diferentes complicaciones."""
    print("Calculando características adicionales...")

    # Características de insulina
    df_modelo = pd.merge(
        df_modelo,
        dataframes["rand_info"][["PtID", "TotDlyIns"]],
        on="PtID",
        how="left",
    )

    # Calcular si usa bomba de insulina (Is_Pump_User)
    # La información está en la tabla screening en la columna InsModPump
    df_screening_pump = dataframes["screening"][["PtID", "InsModPump"]].copy()
    df_screening_pump["Is_Pump_User"] = (
        df_screening_pump["InsModPump"]
        .astype(str)
        .str.lower()
        .str.contains("pump", na=False)
    ).astype(int)
    df_modelo = pd.merge(
        df_modelo, df_screening_pump[["PtID", "Is_Pump_User"]], on="PtID", how="left"
    )

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

    return df_modelo


def seleccionar_caracteristicas_por_complicacion(
    df_modelo, tipo_complicacion, col_objetivo
):
    """
    Selecciona las características relevantes según el tipo de complicación.
    """
    feature_configs = get_feature_configs()

    if tipo_complicacion not in feature_configs:
        print(
            f"ADVERTENCIA: Configuración no encontrada para '{tipo_complicacion}'. Usando configuración básica."
        )
        # Configuración básica por defecto
        columnas_seleccionadas = [
            "PtID",
            "Age",
            "Sex",
            "Duration_of_Diabetes",
            "IMC",
            "Glucose_Mean",
            "Time_In_Range_70_180",
            col_objetivo,
        ]
    else:
        config = feature_configs[tipo_complicacion]
        columnas_seleccionadas = []

        # Combinar todas las categorías de características
        for categoria, caracteristicas in config.items():
            if caracteristicas:  # Solo agregar si la lista no está vacía
                columnas_seleccionadas.extend(caracteristicas)

        # Agregar la columna objetivo
        columnas_seleccionadas.append(col_objetivo)

        # Eliminar duplicados y mantener el orden
        columnas_seleccionadas = list(dict.fromkeys(columnas_seleccionadas))

    # Filtrar solo las columnas que existen en el dataframe
    columnas_disponibles = [
        col for col in columnas_seleccionadas if col in df_modelo.columns
    ]
    columnas_faltantes = [
        col for col in columnas_seleccionadas if col not in df_modelo.columns
    ]

    if columnas_faltantes:
        print(
            f"ADVERTENCIA: Las siguientes características no están disponibles: {columnas_faltantes}"
        )

    print(
        f"Características seleccionadas para {tipo_complicacion}: {len(columnas_disponibles)}"
    )
    print(f"Lista de características: {columnas_disponibles}")

    return df_modelo[columnas_disponibles]


def preparar_dataset_complicacion(dataframes, config):
    """
    Función genérica para preparar un dataset enriquecido para una complicación específica
    con características personalizadas según el tipo de complicación.
    """
    nombre_col_objetivo = config["col_objetivo"]
    terminos_busqueda = config["terminos"]
    nombre_archivo_salida = config["archivo_salida"]
    tipo_complicacion = config["tipo"]  # Nuevo parámetro

    print(f"\n--- Preparando dataset para: {tipo_complicacion.upper()} ---")

    # 1. Crear variable objetivo y tabla base
    df_modelo = dataframes["roster"][["PtID", "AgeAsofEnrollDt"]].copy()
    mascara = dataframes["condiciones"]["MedicalCondition"].str.contains(
        "|".join(terminos_busqueda), case=False, na=False
    )
    ptids_con_complicacion = dataframes["condiciones"][mascara]["PtID"].unique()
    df_modelo[nombre_col_objetivo] = (
        df_modelo["PtID"].isin(ptids_con_complicacion).astype(int)
    )

    # 2. Unir características CGM
    df_cgm_features = calcular_metricas_cgm(dataframes["cgm"])
    df_modelo = pd.merge(df_modelo, df_cgm_features, on="PtID", how="left")

    # 3. Añadir características demográficas
    df_screening_subset = dataframes["screening"][["PtID", "DiagAge", "Sex"]].copy()
    df_modelo = pd.merge(df_modelo, df_screening_subset, on="PtID", how="left")

    # 4. Calcular IMC
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

    # 5. Calcular edad y duración de diabetes
    df_modelo.rename(columns={"AgeAsofEnrollDt": "Age"}, inplace=True)
    df_modelo["Duration_of_Diabetes"] = df_modelo["Age"] - df_modelo["DiagAge"]

    # 6. Añadir característica de hipertensión
    ptids_con_hipertension = dataframes["condiciones"][
        dataframes["condiciones"]["MedicalCondition"].str.contains(
            "hypertension", case=False, na=False
        )
    ]["PtID"].unique()
    df_modelo["Has_Hypertension"] = (
        df_modelo["PtID"].isin(ptids_con_hipertension).astype(int)
    )

    # 7. Calcular características adicionales específicas
    df_modelo = calcular_caracteristicas_adicionales(df_modelo, dataframes)

    # 8. Seleccionar características según el tipo de complicación
    df_modelo = seleccionar_caracteristicas_por_complicacion(
        df_modelo, tipo_complicacion, nombre_col_objetivo
    )

    # 9. Limpieza final
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

    # 10. Verificación final del tamaño del dataframe
    if len(df_modelo) > 440:
        print(
            f"ALERTA: El número de filas ({len(df_modelo)}) es mayor al esperado. Revisa las uniones (merges)."
        )
        df_modelo.drop_duplicates(subset=["PtID"], keep="last", inplace=True)
        print(f"Número de filas corregido a: {len(df_modelo)}")

    # 11. Guardar dataset
    df_modelo.to_csv(nombre_archivo_salida, index=False)
    print(f"Dataset para {tipo_complicacion} guardado en '{nombre_archivo_salida}'")
    print(f"Filas: {len(df_modelo)}, Columnas: {len(df_modelo.columns)}")
    print(
        f"Distribución de la variable objetivo: {df_modelo[nombre_col_objetivo].value_counts().to_dict()}"
    )

    return df_modelo


def mostrar_resumen_configuraciones():
    """Muestra un resumen de las configuraciones de características para cada complicación."""
    print("\n" + "=" * 80)
    print("RESUMEN DE CONFIGURACIONES DE CARACTERÍSTICAS POR COMPLICACIÓN")
    print("=" * 80)

    feature_configs = get_feature_configs()

    for complicacion, config in feature_configs.items():
        print(f"\n🏥 {complicacion.upper().replace('_', ' ')}")
        print("-" * 50)

        total_features = 0
        for categoria, caracteristicas in config.items():
            if caracteristicas:
                print(
                    f"  📊 {categoria.replace('_', ' ').title()}: {len(caracteristicas)} características"
                )
                total_features += len(caracteristicas)
                # Mostrar TODAS las características
                for caracteristica in caracteristicas:
                    if caracteristica != "PtID":
                        print(f"      • {caracteristica}")

        features_sin_ptid = total_features - (
            1 if "PtID" in sum(config.values(), []) else 0
        )
        print(f"  📈 Total de características: {features_sin_ptid}")
        print(
            f"  📈 Total de columnas del dataset: {total_features + 1}"
        ) 

    print("\n" + "=" * 80)


def main():
    """Ejecuta el pipeline completo para generar todos los datasets."""
    print("🔧 INICIANDO GENERACIÓN MODULAR DE DATASETS")

    # Mostrar configuraciones disponibles
    mostrar_resumen_configuraciones()

    ruta_base_datos = "../data/raw/datatables/"

    dataframes = cargar_datos_base(ruta_base_datos)

    if dataframes:
        complicaciones = get_complicaciones_config()

        for comp_config in complicaciones:
            preparar_dataset_complicacion(dataframes, comp_config)

        print(f"\n✅ TODOS LOS DATASETS GENERADOS EXITOSAMENTE")
    else:
        print("❌ Error al cargar los datos base.")


def generar_dataset_especifico(tipo_complicacion):
    """
    Genera un dataset específico para una sola complicación.
    Útil para pruebas rápidas o cuando solo necesitas una complicación.
    """
    print(f"🎯 GENERANDO DATASET ESPECÍFICO PARA: {tipo_complicacion.upper()}")

    ruta_base_datos = "../data/raw/datatables/"
    dataframes = cargar_datos_base(ruta_base_datos)

    if not dataframes:
        print("❌ Error al cargar los datos base.")
        return None

    # Buscar la configuración para la complicación específica
    complicaciones = get_complicaciones_config()

    config_encontrada = None
    for config in complicaciones:
        if config["tipo"] == tipo_complicacion:
            config_encontrada = config
            break

    if not config_encontrada:
        print(f"❌ Configuración no encontrada para '{tipo_complicacion}'")
        print(f"Tipos disponibles: {[c['tipo'] for c in complicaciones]}")
        return None

    # Mostrar configuración específica
    feature_configs = get_feature_configs()
    if tipo_complicacion in feature_configs:
        config_features = feature_configs[tipo_complicacion]
        print(f"\n📋 Configuración de características para {tipo_complicacion}:")
        for categoria, caracteristicas in config_features.items():
            if caracteristicas:
                print(f"  • {categoria}: {len(caracteristicas)} características")

    # Generar el dataset
    df_resultado = preparar_dataset_complicacion(dataframes, config_encontrada)

    if df_resultado is not None:
        print(f"✅ Dataset generado exitosamente para {tipo_complicacion}")
        return df_resultado
    else:
        print(f"❌ Error al generar dataset para {tipo_complicacion}")
        return None


if __name__ == "__main__":

    # Opción 1: Generar todos los datasets
    main()

    # Opción 2: Generar solo un dataset específico
    # generar_dataset_especifico("retinopathy")

    # Opción 3: Solo mostrar las configuraciones disponibles
    # mostrar_resumen_configuraciones()
