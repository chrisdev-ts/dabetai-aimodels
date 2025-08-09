
# dabetai AI Models - Módulo de inteligencia artificial para predicción de complicaciones diabéticas

Módulo de machine learning que implementa los modelos predictivos para las complicaciones diabéticas tipo 1, integrados en la plataforma dabetai.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python version">
  <img src="https://img.shields.io/badge/scikit-learn-1.3-blue?logo=scikitlearn" alt="scikit-learn version">
  <img src="https://img.shields.io/badge/LightGBM-3.x-green?logo=lightgbm" alt="LightGBM version">
  <img src="https://img.shields.io/badge/XGBoost-1.6-orange?logo=xgboost" alt="XGBoost version">
  <img src="https://img.shields.io/badge/joblib-1.x-yellow" alt="joblib version">
</p>

---

## 🤖 ¿Qué es dabetai AI Models?

**dabetai AI Models** contiene los pipelines completos para entrenamiento, evaluación y serialización de modelos de machine learning enfocados en predecir:

- Retinopatía diabética  
- Nefropatía diabética  
- Neuropatía diabética  
- Pie diabético  

Los modelos se basan en datos clínicos y biométricos del estudio IOBP2 y están optimizados con técnicas avanzadas como balanceo de clases, optimización de hiperparámetros y validación cruzada.

---

## ✨ Funcionalidades principales

- Preparación modular y automatizada de datasets específicos por complicación  
- Experimentación comparativa con múltiples algoritmos (Regresión Logística, Random Forest, LightGBM, XGBoost, SVM, AdaBoost)  
- Optimización de hiperparámetros mediante Grid Search  
- Entrenamiento final y serialización de modelos  
- Generación automática de reportes y visualizaciones (ROC, matrices de confusión, importancia de características)  

---

## 🛠 Tecnologías

- **Python 3.11+**  
- **scikit-learn**  
- **LightGBM**  
- **XGBoost**  
- **joblib** para serialización  
- **imbalanced-learn** para balanceo de clases (SMOTE)  
- **matplotlib** y **seaborn** para visualización  

---

## ⚡ Instalación rápida

### Prerrequisitos

- Python 3.11+  
- pip  

### Pasos

1. **Clonar repositorio**

```bash
git clone https://github.com/chrisdev-ts/dabetai-aimodels.git
cd dabetai-aimodels
````

2. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

---

## 📂 Estructura del proyecto

```
dabetai-aimodels/
├── scripts/
│   ├── 01_prepare_datasets.py
│   ├── 02_run_experiments.py
│   └── 03_finalize_model.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
└── requirements.txt
```

---

## 📚 Guía de uso

### 1. Preparar datasets

```bash
python scripts/01_prepare_datasets.py
```

### 2. Ejecutar experimentos

```bash
python scripts/02_run_experiments.py
```

### 3. Finalizar modelos

```bash
python scripts/03_finalize_model.py
```

---

## 🩺 Datos requeridos

Los datos se basan en el estudio **IOBP2 (In Control)**. Deben colocarse en `data/raw/datatables/` con los archivos específicos. Consulta el archivo `CITATION.md` para más detalles sobre atribución y uso responsable.

---

## 🏗 Ecosistema dabetai: nuestros repositorios

dabetai está compuesto por múltiples repositorios especializados:

| Repositorio                                                             | Propósito                   | Estado          |
| ----------------------------------------------------------------------- | --------------------------- | --------------- |
| **[dabetai-mobileapp](https://github.com/Fermin-Cardenas/dabetai-mobileapp)** | App para pacientes          | ✅ En desarrollo |
| **[dabetai-webapp](https://github.com/chrisdev-ts/dabetai-webapp)**     | App web para médicos        | ✅ En desarrollo |
| **[dabetai-aiapi](https://github.com/aleor25/dabetai-aiapi)**           | API de IA y predicciones    | ✅ En desarrollo |
| **[dabetai-aimodels](https://github.com/chrisdev-ts/dabetai-aimodels)** | Modelos de machine learning | ✅ En desarrollo |
| **[dabetai-landing](https://github.com/chrisdev-ts/dabetai-landing)**   | Página de aterrizaje        | ✅ En desarrollo |
| **[dabetai-api](https://github.com/chrisdev-ts/dabetai-api)**                                                         | API principal del backend   | ✅ En desarrollo |

---

## 🤝 Colaboración interna

Seguimos convenciones específicas para mantener consistencia - consulta [CONTRIBUTING.MD](CONTRIBUTING.MD).

---

## 🤝 Reconocimientos

Este proyecto fue desarrollado por el equipo de autores:

* Cardenas Cabal Fermín
* Ortiz Pérez Alejandro
* Serrano Puertos Jorge Christian

Con la asesoría y guía conceptual de:

* Guarneros Nolasco Luis Rolando
* Cruz Ramos Nancy Aracely

Y con el apoyo académico de la

* Universidad Tecnológica del Centro de Veracruz