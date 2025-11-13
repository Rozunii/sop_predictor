# Predictor de Síndrome de Ovario Poliquístico (SOP)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Análisis estadístico de biomarcadores y factores asociados al SOP usando Machine Learning

Sistema de predicción basado en Gradient Boosting para identificar probabilidad de Síndrome de Ovario Poliquístico (PCOS) a partir de variables clínicas, hormonales y de estilo de vida.

**[Documentación Completa](#estructura-del-proyecto)** | **[Paper/Presentación](#referencias)**

---

## Tabla de Contenidos

- [Características](#-características)
- [Resultados](#-resultados)
- [Tecnologías](#️-tecnologías)
- [Instalación](#-instalación-rápida)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Metodología](#-metodología)
- [Equipo](#-equipo)
- [Referencias](#-referencias)
- [Licencia](#-licencia)

---

## Características

### Modelo Predictivo
- **89.81% de accuracy** en predicción de SOP
- **AUC-ROC de 0.9542** (poder discriminativo excelente)
- Gradient Boosting optimizado con 18 variables clínicas
- Validación cruzada de 5 folds

### Análisis Estadístico Completo
- Análisis exploratorio de 541 pacientes
- Pruebas inferenciales (Mann-Whitney, Chi-cuadrado)
- Identificación de variables discriminantes
- Análisis de correlaciones (Spearman)

### Aplicación Web Interactiva
- Interfaz intuitiva desarrollada en Streamlit
- 3 páginas: Predicción, Referencias Médicas, Info del Proyecto
- 10+ referencias científicas con DOI verificado
- Visualizaciones interactivas con Plotly

### Rigor Científico
- Referencias basadas en criterios de Rotterdam
- Guías internacionales 2023 sobre SOP
- Valores normales validados por literatura médica
- Disclaimer apropiado para uso educativo

---

## Resultados

### Comparación de Modelos

| Modelo | Accuracy | AUC-ROC | Precision | Recall |
|--------|----------|---------|-----------|--------|
| **Gradient Boosting** | **89.81%** | **0.9542** | **89%** | **90%** |
| Random Forest | 88.89% | 0.9468 | 88% | 89% |
| SVM | 88.89% | 0.9441 | 89% | 89% |
| Logistic Regression | 87.04% | 0.9282 | 87% | 86% |

### Top 5 Variables Más Importantes

1. **Follicle_R (47.6%)** - Número de folículos en ovario derecho
2. **Hair_growth (7.4%)** - Hirsutismo (crecimiento excesivo de vello)
3. **Weight_gain (7.1%)** - Aumento de peso inexplicable
4. **Skin_darkening (5.9%)** - Acantosis nigricans
5. **Endometrium (4.5%)** - Grosor endometrial

### Hallazgos Clínicos

- **15 de 27 variables continuas** son estadísticamente significativas (p < 0.05)
- **6 de 7 síntomas** altamente significativos para diagnóstico
- **AMH** confirmado como biomarcador clave (p = 6.00e-08)
- Variables ecográficas tienen el mayor poder discriminativo

---

## Tecnologías

### Backend & Machine Learning
- Python 3.8+
- scikit-learn 1.3+ (Gradient Boosting, Random Forest, SVM)
- pandas, numpy (manipulación de datos)
- scipy (análisis estadístico)

### Frontend & Visualización
- Streamlit 1.28+ (aplicación web)
- Plotly 5.17+ (visualizaciones interactivas)
- matplotlib, seaborn (gráficos)

### Deployment
- Git & GitHub
- Jupyter Notebook (análisis)

---

## Instalación Rápida

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/predictor-sop.git
cd predictor-sop
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app_pcos_streamlit.py
```

La aplicación se abrirá en: `http://localhost:8501`

---

## Uso

### Aplicación Web

1. **Página de Predicción**: Ingresa las 18 variables clínicas del paciente
2. **Referencias Médicas**: Consulta valores normales y literatura científica
3. **Info del Proyecto**: Metodología, equipo y resultados

### Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

Incluye:
- Análisis exploratorio completo
- Pruebas de hipótesis
- Entrenamiento de modelos
- Validación y métricas
- Visualizaciones

### API del Modelo (Ejemplo)

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load('modelos/modelo_gb_pcos.pkl')
scaler = joblib.load('modelos/scaler_pcos.pkl')

# Preparar datos
datos = {
    'Follicle_R': 15,
    'Hair_growth': 1,
    'Weight_gain': 1,
    'AMH': 7.5,
    # ... resto de variables
}

df = pd.DataFrame([datos])
df = df[feature_order]  # Orden específico
X_scaled = scaler.transform(df)

# Predecir
probabilidad = modelo.predict_proba(X_scaled)[0][1]
print(f"Probabilidad de SOP: {probabilidad:.1%}")
```

---

## Estructura del Proyecto

```
predictor-sop/
│
├── 📄 README.md                      # Este archivo
├── 📄 requirements.txt               # Dependencias
├── 📄 .gitignore                     # Archivos ignorados
│
├── 📓 main.ipynb                     # Notebook principal con análisis
│
├── 🌐 app_pcos_streamlit.py         # Aplicación web Streamlit
│
├── 📂 modelos/                       # Modelos entrenados
│   ├── modelo_gb_pcos.pkl           # Gradient Boosting
│   ├── scaler_pcos.pkl              # StandardScaler
│   └── model_info.pkl               # Metadata del modelo
│
├── 📂 datos/                         # Dataset (opcional)
│   └── PCOS_data_1.xlsx             # Dataset original
│
├── 📂 scripts/                       # Scripts auxiliares
│   ├── generar_modelo_ejemplo.py    # Generar modelo de prueba
│   └── verificar_modelo.py          # Verificar compatibilidad
│
├── 📂 docs/                          # Documentación
│   ├── GUIA_RAPIDA.md               # Guía de inicio rápido
│   ├── REFERENCIAS_MEDICAS.md       # Referencias científicas
│   └── presentacion.pdf             # Presentación del proyecto
│
└── 📂 visualizaciones/               # Gráficos generados
    ├── importancia_variables.png
    ├── curvas_roc.png
    └── matriz_confusion.png
```

---

## Metodología

### 1. Análisis Exploratorio
- Verificación de datos (541 pacientes, 42 variables)
- Identificación de valores faltantes (solo 3 valores)
- Análisis de distribuciones y outliers
- Visualizaciones descriptivas

### 2. Análisis Inferencial
- **Pruebas para variables continuas**: Mann-Whitney U (no paramétricas)
- **Pruebas para variables categóricas**: Chi-cuadrado
- **Análisis de correlaciones**: Spearman (no asume normalidad)
- **Nivel de significancia**: α = 0.05

### 3. Preprocesamiento
- Escalado con StandardScaler (variables continuas)
- Codificación de variables categóricas (ya en formato binario)
- Split 80/20 (entrenamiento/prueba) con estratificación

### 4. Modelado
- Entrenamiento de 7 modelos diferentes
- Validación cruzada de 5 folds
- Optimización de hiperparámetros
- Selección del mejor modelo: Gradient Boosting

### 5. Evaluación
- Métricas: Accuracy, AUC-ROC, Precision, Recall, F1-Score
- Matriz de confusión
- Curvas de aprendizaje (verificar no overfitting)
- Análisis de importancia de variables

### 6. Interpretabilidad
- Feature importance (sklearn)
- Análisis de errores
- Validación clínica de resultados

---

## Equipo

**Los Poliquísticos** - Proyecto de Probabilidad y Estadística II

- [Bernardo Alejandro Partidas Díaz]
- [Oscar Josue López González]
- [Rodrigo Alonso Castillo Ramírez]
- [Sebastian Sánchez Espinosa]

**Institución**: Centro Universitario de Guadalajara (CUGDL) - Universidad de Guadalajara  
**Asignatura**: Probabilidad y Estadística II  
**Profesora**: Claudia Fabiola  
**Fecha**: Noviembre 2024

---

## Referencias

### Literatura Científica Principal

1. **Teede HJ, et al. (2023).** "Recommendations from the 2023 International Evidence-based Guideline for the Assessment and Management of Polycystic Ovary Syndrome." *Journal of Clinical Endocrinology & Metabolism*, 108(10): 2447-2469. [DOI: 10.1210/clinem/dgad463](https://doi.org/10.1210/clinem/dgad463)

2. **Rotterdam ESHRE/ASRM (2004).** "Revised 2003 consensus on diagnostic criteria and long-term health risks related to polycystic ovary syndrome." *Fertility and Sterility*, 81(1): 19-25. [DOI: 10.1016/j.fertnstert.2003.10.004](https://doi.org/10.1016/j.fertnstert.2003.10.004)

3. **Dewailly D, et al. (2014).** "Diagnosis of polycystic ovary syndrome (PCOS): revisiting the threshold values of follicle count on ultrasound and of the serum AMH level for the definition of polycystic ovaries." *Human Reproduction*, 29(11): 2427-2436. [DOI: 10.1093/humrep/deu234](https://doi.org/10.1093/humrep/deu234)

Ver lista completa en: [docs/REFERENCIAS_MEDICAS.md](docs/REFERENCIAS_MEDICAS.md)

### Dataset

- **Tamaño**: 541 pacientes
- **Variables**: 42 (18 usadas para predicción)
- **Criterio diagnóstico**: Rotterdam 2003

---

## Disclaimer

**IMPORTANTE**: Esta herramienta fue desarrollada con fines **educativos** como parte de un proyecto académico. 

**NO debe utilizarse como herramienta de diagnóstico médico**  
El diagnóstico de SOP debe realizarse por un profesional de la salud calificado

El modelo predictivo se basa en datos históricos y debe ser validado clínicamente antes de cualquier uso en contexto médico real.

## Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Agradecimientos

- Dra. Claudia Fabiola por la asesoría en el proyecto
- Centro Universitario de Guadalajara (CUGDL) - UDG
- Comunidad de código abierto (scikit-learn, Streamlit, etc.)
- Dataset original de Carlos Alberto Fregoso Iturria

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐**

Desarrollado con ❤️ por Los Poliquísticos

</div>
