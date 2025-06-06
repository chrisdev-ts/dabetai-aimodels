import pandas as pd
import numpy as np

def cargar_y_preprocesar_datos_base(ruta_base):
    """Carga y realiza una limpieza inicial en las tablas de datos base."""
    print("Paso 1: Cargando datos base...")
    
    # Definir rutas a los archivos
    archivos = {
        'cgm': f"{ruta_base}IOBP2DeviceCGM.txt",
        'condiciones': f"{ruta_base}IOBP2MedicalCondition.txt",
        'screening': f"{ruta_base}IOBP2DiabScreening.txt",
        'roster': f"{ruta_base}IOBP2PtRoster.txt",
        'hba1c': f"{ruta_base}IOBP2DiabLocalHbA1c.txt",
        'altura_peso': f"{ruta_base}IOBP2HeightWeight.txt"
    }

    # Cargar datos
    dfs = {nombre: pd.read_csv(ruta, delimiter='|', low_memory=False) for nombre, ruta in archivos.items()}
    
    # --- Preprocesamiento de Fechas (con el formato optimizado) ---
    print("Paso 2: Convirtiendo tipos de datos y fechas...")
    dfs['cgm']['DeviceDtTm'] = pd.to_datetime(dfs['cgm']['DeviceDtTm'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    dfs['condiciones']['MedCondDiagDt'] = pd.to_datetime(dfs['condiciones']['MedCondDiagDt'], errors='coerce')
    dfs['screening']['DiagDt'] = pd.to_datetime(dfs['screening']['DiagDt'], errors='coerce')
    dfs['roster']['EnrollDt'] = pd.to_datetime(dfs['roster']['EnrollDt'], errors='coerce')
    
    # Eliminar filas donde la fecha del CGM no se pudo convertir
    dfs['cgm'].dropna(subset=['DeviceDtTm'], inplace=True)
    
    print("Datos base cargados y preprocesados.")
    return dfs

def crear_variable_objetivo_retinopatia(df_condiciones, df_pacientes):
    """Identifica pacientes con retinopatía y crea la variable objetivo."""
    print("Paso 3: Creando la variable objetivo (Retinopathy_Status)...")
    
    df_condiciones['MedicalCondition_lower'] = df_condiciones['MedicalCondition'].str.lower()
    
    terminos_retinopatia = ['retinopathy', 'proliferative diabetic retinopathy', 'non-proliferative diabetic retinopathy']
    mascara_retinopatia = df_condiciones['MedicalCondition_lower'].str.contains('|'.join(terminos_retinopatia), na=False)
    
    ptids_con_retinopatia = df_condiciones[mascara_retinopatia]['PtID'].unique()
    
    # Crear la columna en el DataFrame de pacientes
    df_pacientes['Retinopathy_Status'] = df_pacientes['PtID'].isin(ptids_con_retinopatia).astype(int)
    
    print(f"Pacientes identificados con retinopatía: {len(ptids_con_retinopatia)}")
    print("Distribución de la variable objetivo:")
    print(df_pacientes['Retinopathy_Status'].value_counts())
    
    return df_pacientes

def calcular_metricas_cgm(df_cgm):
    """Calcula métricas de resumen de CGM por paciente."""
    print("Paso 4: Calculando métricas de CGM para cada paciente... (Esto puede tomar un momento)")

    def coefficient_of_variation(series):
        return series.std() / series.mean() if series.mean() != 0 else 0

    df_cgm_features = df_cgm.groupby('PtID')['Value'].agg(
        Glucose_Mean='mean',
        Glucose_Std='std',
        Glucose_CV=coefficient_of_variation,
        Time_Above_180=lambda s: (s > 180).mean() * 100,
        Time_Below_70=lambda s: (s < 70).mean() * 100,
        Time_In_Range_70_180=lambda s: ((s >= 70) & (s <= 180)).mean() * 100
    ).reset_index()
    
    print("Métricas de CGM calculadas exitosamente.")
    return df_cgm_features

def calcular_caracteristicas_demograficas_clinicas(dfs):
    """Calcula y une características como Duración de la Diabetes e IMC."""
    print("Paso 5: Calculando características demográficas y clínicas (Duración DM, IMC)...")
    
    # Unir roster y screening para obtener edad y fecha de diagnóstico
    df_demograficos = pd.merge(dfs['roster'], dfs['screening'], on='PtID', how='left')
    
    # Calcular Duración de la Diabetes en años
    df_demograficos['Duration_of_Diabetes'] = (df_demograficos['EnrollDt'] - df_demograficos['DiagDt']).dt.days / 365.25
    
    # Calcular IMC (necesitamos la medición más reciente por paciente)
    # Primero, estandarizamos unidades (asumiendo que hay mezcla, si no, se puede simplificar)
    df_altura_peso = dfs['altura_peso'].copy()
    df_altura_peso['Weight_kg'] = np.where(df_altura_peso['WeightUnits'] == 'lbs', df_altura_peso['Weight'] * 0.453592, df_altura_peso['Weight'])
    df_altura_peso['Height_m'] = np.where(df_altura_peso['HeightUnits'] == 'in', df_altura_peso['Height'] * 0.0254,
                                       np.where(df_altura_peso['HeightUnits'] == 'cm', df_altura_peso['Height'] / 100, df_altura_peso['Height']))
    
    df_altura_peso['IMC'] = df_altura_peso['Weight_kg'] / (df_altura_peso['Height_m'] ** 2)
    
    # Tomar la última medición de IMC por paciente
    df_imc_final = df_altura_peso.sort_values(by='WeightAssessDt').drop_duplicates('PtID', keep='last')
    
    # Unir HbA1c (tomando la más reciente si hay varias)
    df_hba1c_final = dfs['hba1c'].sort_values(by='HbA1cTestDt').drop_duplicates('PtID', keep='last')
    
    # Unir todo
    df_final = pd.merge(df_demograficos, df_imc_final[['PtID', 'IMC']], on='PtID', how='left')
    df_final = pd.merge(df_final, df_hba1c_final[['PtID', 'HbA1cTestRes']], on='PtID', how='left')
    
    # Seleccionar columnas finales de interés
    columnas_interes = ['PtID', 'AgeAsofEnrollDt', 'Sex', 'Duration_of_Diabetes', 'IMC', 'HbA1cTestRes', 'Retinopathy_Status']
    df_final = df_final[columnas_interes]
    
    print("Características demográficas y clínicas calculadas.")
    return df_final

def main():
    """Función principal para ejecutar el pipeline de preparación de datos."""
    ruta_base_datos = "./datatables/"
    
    # Pasos 1 y 2
    dataframes = cargar_y_preprocesar_datos_base(ruta_base_datos)
    
    # Paso 3
    df_pacientes_con_target = crear_variable_objetivo_retinopatia(
        dataframes['condiciones'], 
        dataframes['roster'].copy() # Usamos una copia de df_roster como base de pacientes
    )
    
    # Paso 4
    df_cgm_features = calcular_metricas_cgm(dataframes['cgm'])
    
    # Paso 5
    # Pasamos el df_roster original y otros a esta función
    df_demograficos_clinicos = calcular_caracteristicas_demograficas_clinicas(dataframes)
    
    # --- Paso Final: Unir Todo en un Dataset para el Modelo ---
    print("\nPaso 6: Uniendo todas las características en el dataset final...")
    
    # Empezamos con la tabla de pacientes que ya tiene la variable objetivo
    dataset_final = pd.merge(df_pacientes_con_target, df_demograficos_clinicos.drop(columns='Retinopathy_Status', errors='ignore'), on='PtID', how='left')
    
    # Añadimos las características de CGM
    dataset_final = pd.merge(dataset_final, df_cgm_features, on='PtID', how='left')
    
    # Eliminar columnas de IDs que ya no necesitamos para el modelo
    dataset_final.drop(columns=['RecID', 'SiteID', 'EnrollDt', 'RandDt', 'VisitSchedStartDt', 'TrtGroup', 'RCTPtStatus', 'TransRandDt', 'TransTrtGroup'], inplace=True, errors='ignore')

    print("\nDataset final creado. Primeras 5 filas:")
    print(dataset_final.head())
    
    print("\nInformación del dataset final:")
    dataset_final.info()
    
    # Opcional: Guardar el dataset final a un archivo CSV
    dataset_final.to_csv("retinopathy_prediction_dataset.csv", index=False)
    print("\nDataset final guardado en 'retinopathy_prediction_dataset.csv'")

# Ejecutar el script
if __name__ == "__main__":
    main()