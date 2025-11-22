import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ====================== CONFIGURACIÓN ======================
st.set_page_config(
    page_title="SOP Predictor | Sistema de Análisis Clínico",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== ESTILOS CSS PROFESIONALES ======================
st.markdown("""
<style>
    /* Importar fuentes profesionales */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Variables de color */
    :root {
        --crimson: #DC143C;
        --light-crimson: #FFB6C1;
        --light-crimson-soft: #FFC0CB;
        --light-crimson-bg: #FFF0F5;
        --dark-crimson: #8B0000;
        --text-primary: #2D2D2D;
        --text-secondary: #666666;
        --text-light: #999999;
        --bg-white: #FFFFFF;
        --bg-gray: #F8F9FA;
        --border-light: #E5E7EB;
    }
    
    /* Reset y base */
    .stApp {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos por defecto de Streamlit */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Tipografía principal */
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--crimson) 0%, var(--light-crimson) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.125rem;
        font-weight: 400;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 0.01em;
    }
    
    /* Headers de sección */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: var(--crimson);
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--light-crimson);
        letter-spacing: -0.01em;
    }
    
    .subsection-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.005em;
    }
    
    /* Tarjetas profesionales */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(220, 20, 60, 0.1), 
                    0 2px 4px -1px rgba(220, 20, 60, 0.06);
        border: 1px solid var(--light-crimson-soft);
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(220, 20, 60, 0.15), 
                    0 4px 6px -2px rgba(220, 20, 60, 0.08);
    }
    
    /* Campos de formulario mejorados */
    .field-container {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 1px solid var(--border-light);
        transition: all 0.2s ease;
    }
    
    .field-container:hover {
        border-color: var(--light-crimson);
        box-shadow: 0 0 0 3px rgba(255, 182, 193, 0.1);
    }
    
    .field-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .field-help {
        font-size: 0.813rem;
        color: var(--text-light);
        margin-top: 0.25rem;
        font-style: italic;
    }
    
    /* Categorías de campos */
    .critical-field {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF0F5 100%);
        border-left: 4px solid var(--crimson);
    }
    
    .important-field {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8FA 100%);
        border-left: 4px solid var(--light-crimson);
    }
    
    .optional-field {
        background: var(--bg-gray);
        border-left: 4px solid var(--border-light);
    }
    
    /* Botones profesionales */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--crimson) 0%, var(--dark-crimson) 100%);
        color: white;
        border: none;
        padding: 0.875rem 2.5rem;
        border-radius: 50px;
        font-size: 1rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(220, 20, 60, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(220, 20, 60, 0.4);
        background: linear-gradient(135deg, var(--dark-crimson) 0%, var(--crimson) 100%);
    }
    
    /* Métricas personalizadas */
    .metric-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(220, 20, 60, 0.08);
        border: 1px solid var(--light-crimson-soft);
        height: 100%;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--crimson);
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Tabs modernos */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid var(--light-crimson-soft);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        letter-spacing: 0.01em;
        color: var(--text-secondary);
        background-color: transparent;
        border: none;
        padding: 0.75rem 0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--crimson);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--crimson);
        border-bottom: 3px solid var(--crimson);
        font-weight: 600;
    }
    
    /* Referencias y citas */
    .reference-box {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8FA 100%);
        border: 1px solid var(--light-crimson-soft);
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .reference-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--crimson) 0%, var(--light-crimson) 100%);
    }
    
    .citation {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: var(--text-light);
        font-style: italic;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-light);
    }
    
    /* Alertas estilizadas */
    .alert-box {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.938rem;
        line-height: 1.6;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #F0FFF4 0%, #DCFCE7 100%);
        border: 1px solid #86EFAC;
        color: #166534;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1px solid #FCD34D;
        color: #92400E;
    }
    
    .alert-error {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        border: 1px solid #FCA5A5;
        color: #991B1B;
    }
    
    .alert-info {
        background: linear-gradient(135deg, var(--light-crimson-bg) 0%, #FFE4E9 100%);
        border: 1px solid var(--light-crimson);
        color: var(--dark-crimson);
    }
    
    /* Sidebar mejorado */
    .css-1d391kg, .st-emotion-cache-1d391kg {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF8FA 100%);
        border-right: 1px solid var(--light-crimson-soft);
    }
    
    /* Radio buttons personalizados */
    .stRadio [role="radiogroup"] {
        gap: 1rem;
    }
    
    .stRadio [data-baseweb="radio"] {
        background-color: white;
        border: 2px solid var(--light-crimson);
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .stRadio [data-baseweb="radio"]:hover {
        background-color: var(--light-crimson-bg);
        transform: translateX(4px);
    }
    
    /* Selectbox mejorado */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid var(--light-crimson);
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--crimson);
        box-shadow: 0 0 0 3px rgba(220, 20, 60, 0.1);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid var(--light-crimson);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--crimson);
        box-shadow: 0 0 0 3px rgba(220, 20, 60, 0.1);
        outline: none;
    }
    
    /* Progress bar personalizado */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--light-crimson) 0%, var(--crimson) 100%);
    }
    
    /* Expander mejorado */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--crimson);
        background-color: var(--light-crimson-bg);
        border: 1px solid var(--light-crimson);
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--light-crimson-soft);
    }
    
    /* División de línea elegante */
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--light-crimson) 50%, transparent 100%);
    }
    
    /* Footer personalizado */
    .custom-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid var(--light-crimson-soft);
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .custom-footer strong {
        color: var(--crimson);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ====================== FUNCIONES ======================

@st.cache_resource
def load_model():
    """Cargar modelo y configuración"""
    try:
        # 1. Obtener la ruta absoluta de donde está ESTE archivo (App2.py)
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construir las rutas completas automáticamente
        ruta_modelo = os.path.join(ruta_base, 'modelos_pcos', 'modelo_gb_pcos.pkl')
        ruta_scaler = os.path.join(ruta_base, 'modelos_pcos', 'scaler_pcos.pkl')
        ruta_info = os.path.join(ruta_base, 'modelos_pcos', 'model_info.pkl')

        # 3. Cargar usando esas rutas construidas
        modelo = joblib.load(ruta_modelo)
        scaler = joblib.load(ruta_scaler)
        model_info = joblib.load(ruta_info)
        
        return modelo, scaler, model_info
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None

def create_risk_gauge(probability):
    """Crear medidor visual de riesgo"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Probabilidad de SOP", 
               'font': {'size': 24, 'family': 'Space Grotesk, sans-serif', 'color': '#DC143C'}},
        number={'suffix': "%", 'font': {'size': 40, 'family': 'Space Grotesk, sans-serif', 'color': '#DC143C'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#FFB6C1"},
            'bar': {'color': "#DC143C", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#FFB6C1",
            'steps': [
                {'range': [0, 30], 'color': '#FFE4E9'},
                {'range': [30, 60], 'color': '#FFB6C1'},
                {'range': [60, 100], 'color': '#FF8FA3'}
            ],
            'threshold': {
                'line': {'color': "#8B0000", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Inter, sans-serif'}
    )
    
    return fig

def predecir_pcos(datos, modelo, scaler):
    """Realizar predicción de SOP"""
    df_input = pd.DataFrame([datos])
    
    if 'features' in st.session_state.model_info:
        df_input = df_input[st.session_state.model_info['features']]
    
    probabilidad = modelo.predict_proba(df_input)[0][1]
    
    if probabilidad < 0.3:
        nivel = "BAJO"
        color = "#22C55E"
        mensaje = "Baja probabilidad de SOP"
        recomendacion = "Mantén un estilo de vida saludable y controles ginecológicos anuales. Recuerda que esta herramienta es con fines educativos NO reemplaza los diagnosticos medicos ni es 100% segura."
    elif probabilidad < 0.6:
        nivel = "MODERADO"
        color = "#F59E0B"
        mensaje = "Riesgo moderado de SOP"
        recomendacion = "Se recomienda consultar con un ginecólogo para una evaluación completa. Recuerda que esta herramienta es con fines educativos NO reemplaza los diagnosticos medicos ni es 100% segura."
    else:
        nivel = "ALTO"
        color = "#DC143C"
        mensaje = "Alta probabilidad de SOP"
        recomendacion = "Es importante consultar pronto con un especialista en endocrinología o ginecología. Recuerda que esta herramienta es con fines educativos NO reemplaza los diagnosticos medicos ni es 100% segura."
    
    return {
        'probabilidad': probabilidad,
        'nivel': nivel,
        'color': color,
        'mensaje': mensaje,
        'recomendacion': recomendacion
    }

# ====================== APLICACIÓN PRINCIPAL ======================

# Inicializar estado
if 'modelo' not in st.session_state:
    modelo, scaler, model_info = load_model()
    st.session_state.modelo = modelo
    st.session_state.scaler = scaler
    st.session_state.model_info = model_info or {}

# Header principal
st.markdown('<h1 class="main-title">Sistema de Predicción SOP</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Análisis Estadístico de Biomarcadores y Factores Clínicos</p>', unsafe_allow_html=True)

# Navegación
tab1, tab2, tab3 = st.tabs(["ANÁLISIS PREDICTIVO", "REFERENCIAS MÉDICAS", "INFORMACIÓN"])

# ====================== TAB 1: ANÁLISIS ======================
with tab1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #DC143C; font-family: 'Space Grotesk', sans-serif;">Paso 1</h3>
            <p style="color: #666;">Complete los síntomas observables</p>
        </div>
        """, unsafe_allow_html=True)
    with col_info2:
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #DC143C; font-family: 'Space Grotesk', sans-serif;">Paso 2</h3>
            <p style="color: #666;">Agregue estudios médicos (opcional)</p>
        </div>
        """, unsafe_allow_html=True)
    with col_info3:
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #DC143C; font-family: 'Space Grotesk', sans-serif;">Paso 3</h3>
            <p style="color: #666;">Obtenga su evaluación de riesgo</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("formulario_sop"):
        # SECCIÓN 1: SÍNTOMAS OBSERVABLES
        st.markdown('<h2 class="section-header">Síntomas Observables</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="field-container critical-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Aumento de Peso</label>', unsafe_allow_html=True)
            weight_gain = st.selectbox(
                "",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                key="weight_gain",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">Sin causa aparente</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="field-container critical-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Hirsutismo</label>', unsafe_allow_html=True)
            hair_growth = st.selectbox(
                "",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                key="hair_growth",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">Vello excesivo facial/corporal</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="field-container critical-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Acantosis Nigricans</label>', unsafe_allow_html=True)
            skin_darkening = st.selectbox(
                "",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                key="skin_darkening",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">Manchas oscuras en piel</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="field-container critical-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Alopecia</label>', unsafe_allow_html=True)
            hair_loss = st.selectbox(
                "",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                key="hair_loss",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">Pérdida de cabello</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="field-container critical-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Acné Persistente</label>', unsafe_allow_html=True)
            pimples = st.selectbox(
                "",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                key="pimples",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">Resistente a tratamiento</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="field-container important-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Consumo Comida Rápida</label>', unsafe_allow_html=True)
            fast_food = st.selectbox(
                "",
                options=[0, 1],
                format_func=lambda x: "Bajo" if x == 0 else "Frecuente",
                key="fast_food",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">Frecuencia semanal</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 2: DATOS ANTROPOMÉTRICOS
        st.markdown('<h2 class="section-header">Datos Antropométricos y Ciclo Menstrual</h2>', unsafe_allow_html=True)
        
        col4, col5, col6, col7 = st.columns(4)
        
        with col4:
            st.markdown('<div class="field-container important-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Edad (años)</label>', unsafe_allow_html=True)
            age = st.number_input("", min_value=15, max_value=50, value=28, key="age", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="field-container important-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">IMC (kg/m²)</label>', unsafe_allow_html=True)
            bmi = st.number_input("", min_value=15.0, max_value=45.0, value=23.0, step=0.1, key="bmi", label_visibility="collapsed")
            st.markdown('<p class="field-help">Normal: 18.5-24.9</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown('<div class="field-container important-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Duración Sangrado (días)</label>', unsafe_allow_html=True)
            cycle_length = st.number_input("", min_value=0, max_value=12, value=5, key="cycle_length", label_visibility="collapsed")
            st.markdown('<p class="field-help">Normal: 3-7 días</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col7:
            st.markdown('<div class="field-container important-field">', unsafe_allow_html=True)
            st.markdown('<label class="field-label">Regularidad Ciclo</label>', unsafe_allow_html=True)
            cycle_ri = st.selectbox(
                "",
                options=[2, 4],
                format_func=lambda x: "Regular" if x == 2 else "Irregular",
                key="cycle_ri",
                label_visibility="collapsed"
            )
            st.markdown('<p class="field-help">21-35 días = regular</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 3: ESTUDIOS MÉDICOS
        st.markdown('<h2 class="section-header">Estudios Médicos</h2>', unsafe_allow_html=True)
        
        tiene_estudios = st.checkbox("Tengo resultados de laboratorio disponibles", value=False)
        
        if tiene_estudios:
            col8, col9, col10 = st.columns(3)
            
            with col8:
                st.markdown('<h3 class="subsection-header">Ultrasonido</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">Folículos Ovario Derecho</label>', unsafe_allow_html=True)
                follicle_r = st.number_input("", min_value=0, max_value=30, value=5, key="follicle_r", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: 2-10, SOP: >12</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">Grosor Endometrial (mm)</label>', unsafe_allow_html=True)
                endometrium = st.number_input("", min_value=3.0, max_value=20.0, value=8.0, step=0.1, key="endometrium", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: 7-14 mm</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">Tamaño Promedio Folículos (mm)</label>', unsafe_allow_html=True)
                avg_f_size_l = st.number_input("", min_value=2.0, max_value=25.0, value=10.0, step=0.1, key="avg_f_size_l", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col9:
                st.markdown('<h3 class="subsection-header">Perfil Hormonal</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">AMH (ng/mL)</label>', unsafe_allow_html=True)
                amh = st.number_input("", min_value=0.0, max_value=20.0, value=3.0, step=0.1, key="amh", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: <4.0, SOP: >4.7</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">FSH (mIU/mL)</label>', unsafe_allow_html=True)
                fsh = st.number_input("", min_value=0.0, max_value=25.0, value=5.0, step=0.1, key="fsh", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: 3-10 mIU/mL</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">Ratio FSH/LH</label>', unsafe_allow_html=True)
                fsh_lh = st.number_input("", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="fsh_lh", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: >2, SOP: <1</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col10:
                st.markdown('<h3 class="subsection-header">Otros Parámetros</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">Frecuencia Cardíaca (bpm)</label>', unsafe_allow_html=True)
                pulse_rate = st.number_input("", min_value=50, max_value=120, value=72, key="pulse_rate", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: 60-100 bpm</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-container optional-field">', unsafe_allow_html=True)
                st.markdown('<label class="field-label">Hemoglobina (g/dL)</label>', unsafe_allow_html=True)
                hb = st.number_input("", min_value=8.0, max_value=18.0, value=12.0, step=0.1, key="hb", label_visibility="collapsed")
                st.markdown('<p class="field-help">Normal: 12-16 g/dL</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <strong>Sin estudios médicos disponibles</strong><br>
                Se utilizarán valores de referencia normales para el análisis. 
                La evaluación se basará principalmente en los síntomas clínicos reportados.
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            follicle_r = 5
            endometrium = 8.0
            amh = 3.0
            fsh_lh = 1.0
            fsh = 5.0
            avg_f_size_l = 10.0
            pulse_rate = 72
            hb = 12.0
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("GENERAR ANÁLISIS PREDICTIVO", use_container_width=True)
        
        if submitted:
            if st.session_state.modelo is None:
                st.markdown('<div class="alert-box alert-error">', unsafe_allow_html=True)
                st.markdown("Sistema no disponible. Por favor, verifique la configuración.", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Preparar datos
                datos = {
                    'Follicle_R': follicle_r,
                    'Cycle_length': cycle_length,
                    'AMH': amh,
                    'BMI': bmi,
                    'Age': age,
                    'Pulse_rate': pulse_rate,
                    'Endometrium': endometrium,
                    'FSH/LH': fsh_lh,
                    'FSH': fsh,
                    'Avg_F_size_L': avg_f_size_l,
                    'Hb': hb,
                    'Weight_gain': weight_gain,
                    'Hair_growth': hair_growth,
                    'Skin_darkening': skin_darkening,
                    'Hair_loss': hair_loss,
                    'Pimples': pimples,
                    'Fast_food': fast_food,
                    'Cycle_RI': cycle_ri
                }
                
                # Predicción
                resultado = predecir_pcos(datos, st.session_state.modelo, st.session_state.scaler)
                
                st.markdown('<h2 class="section-header">Resultados del Análisis</h2>', unsafe_allow_html=True)
                
                # Métricas principales
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Probabilidad SOP</p>
                        <p class="metric-value">{resultado['probabilidad']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Nivel de Riesgo</p>
                        <p class="metric-value" style="color: {resultado['color']};">{resultado['nivel']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Estado</p>
                        <p class="metric-value" style="font-size: 1.5rem;">{resultado['mensaje']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gráfico de probabilidad
                st.markdown("<br>", unsafe_allow_html=True)
                fig = create_risk_gauge(resultado['probabilidad'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Recomendaciones
                st.markdown('<h3 class="subsection-header">Recomendaciones Clínicas</h3>', unsafe_allow_html=True)
                
                
                st.markdown(f"<p style='font-size: 1.1rem; line-height: 1.8; color: #2D2D2D;'>{resultado['recomendacion']}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Síntomas detectados
                if any([weight_gain, hair_growth, skin_darkening, hair_loss, pimples, cycle_ri == 4]):
                    st.markdown('<h3 class="subsection-header">Síntomas Detectados</h3>', unsafe_allow_html=True)
                    
                    sintomas = []
                    if weight_gain: sintomas.append("Aumento de peso inexplicable")
                    if hair_growth: sintomas.append("Hirsutismo")
                    if skin_darkening: sintomas.append("Acantosis nigricans")
                    if hair_loss: sintomas.append("Alopecia androgénica")
                    if pimples: sintomas.append("Acné persistente")
                    if cycle_ri == 4: sintomas.append("Ciclo menstrual irregular")
                    
                    cols_sint = st.columns(2)
                    for i, sintoma in enumerate(sintomas):
                        with cols_sint[i % 2]:
                            st.markdown(f"""
                            <div style='padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid #DC143C; background: #FFF5F7;'>
                                <span style='color: #2D2D2D; font-weight: 500;'>{sintoma}</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Advertencia legal
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown('<div class="alert-box alert-warning">', unsafe_allow_html=True)
                st.markdown("""
                <strong>AVISO MÉDICO IMPORTANTE</strong><br>
                Esta herramienta proporciona únicamente una evaluación de riesgo basada en análisis estadístico 
                y no constituye un diagnóstico médico. El diagnóstico definitivo de SOP debe ser realizado 
                por un profesional médico cualificado mediante evaluación clínica completa.
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ====================== TAB 2: REFERENCIAS ======================
with tab2:
    st.markdown('<h2 class="section-header">Referencias Médicas y Criterios Diagnósticos</h2>', unsafe_allow_html=True)
    
    # Criterios de Rotterdam
    st.markdown('<h3 class="subsection-header">Criterios de Rotterdam (2003)</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    **Diagnóstico de SOP requiere 2 de 3 criterios:**
    
    1. **Oligoovulación o anovulación**
       - Ciclos menstruales > 35 días
       - Menos de 8 ciclos por año
    
    2. **Hiperandrogenismo**
       - Clínico: Hirsutismo (Ferriman-Gallwey ≥ 8), acné, alopecia
       - Bioquímico: Andrógenos elevados
    
    3. **Morfología ovárica poliquística**
       - ≥12 folículos (2-9 mm) por ovario
       - Volumen ovárico > 10 cm³
    
    <p class="citation">Rotterdam ESHRE/ASRM Consensus Workshop Group (2004). Fertility and Sterility, 81(1): 19-25.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Biomarcadores
    st.markdown('<h3 class="subsection-header">Biomarcadores Principales</h3>', unsafe_allow_html=True)
    
    biomarcadores = pd.DataFrame({
        'Biomarcador': ['AMH', 'Relación LH/FSH', 'Testosterona Total', 'DHEA-S', 'Índice FAI'],
        'Valor Normal': ['1.0-4.0 ng/mL', '< 2', '< 70 ng/dL', '35-430 μg/dL', '< 4.5'],
        'SOP': ['> 4.7 ng/mL', '> 2-3', '> 70 ng/dL', 'Elevado', '> 4.5'],
        'Sensibilidad': ['82%', '40%', '70%', '35%', '75%'],
        'Especificidad': ['88%', '95%', '75%', '90%', '85%']
    })
    
    st.dataframe(biomarcadores, use_container_width=True, hide_index=True)
    
    # Referencias bibliográficas
    st.markdown('<h3 class="subsection-header">Bibliografía Esencial</h3>', unsafe_allow_html=True)
    
    referencias = [
        "Teede HJ, et al. (2023). International Evidence-based Guideline for PCOS. J Clin Endocrinol Metab.",
        "Dewailly D, et al. (2014). AMH as diagnostic marker for PCOS. Hum Reprod.",
        "Yildiz BO, et al. (2010). Prevalence and phenotypes of PCOS. Hum Reprod.",
        "Fraser IS, et al. (2011). FIGO recommendations on uterine bleeding. Semin Reprod Med."
    ]
    
    for ref in referencias:
        st.markdown(f"""
        <div style='padding: 0.5rem; margin: 0.5rem 0; border-left: 2px solid #FFB6C1; background: #FFF8FA;'>
            <span style='color: #666; font-size: 0.9rem;'>{ref}</span>
        </div>
        """, unsafe_allow_html=True)

# ====================== TAB 3: INFORMACIÓN ======================
with tab3:
    st.markdown('<h2 class="section-header">Información del Proyecto</h2>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown("""
        <h3 style="color: #DC143C; font-family: 'Space Grotesk', sans-serif;">Equipo de Desarrollo</h3>
        
        **Los Poliquísticos**
        - Bernardo Alejandro Partidas Díaz
        - Oscar Josue López González  
        - Rodrigo Alonso Castillo Ramírez
        - Sebastian Sánchez Espinosa
        
        <hr style='margin: 1.5rem 0; opacity: 0.3;'>
        
        **Centro Universitario de Guadalajara (CUGDL)**  
        Universidad de Guadalajara  
        
        Asignatura: Probabilidad y Estadística II  
        Profesora: Claudia Fabiola
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <h3 style="color: #DC143C; font-family: 'Space Grotesk', sans-serif;">Métricas del Modelo</h3>
        
        **Gradient Boosting Classifier**
        - Accuracy: 87.04%
        - AUC-ROC: 0.9491
        - Sensibilidad: 85.71%
        - Especificidad: 87.67%
        
        **Dataset**
        - 541 pacientes
        - 42 variables totales
        - 18 características predictoras
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-footer">
        <strong>SOP Predictor v3.0</strong> | Proyecto Académico CUGDL<br>
        Universidad de Guadalajara | Noviembre 2025<br>
        <span style='font-size: 0.75rem;'>Desarrollado con fines educativos y de investigación</span>
    </div>
    """, unsafe_allow_html=True)