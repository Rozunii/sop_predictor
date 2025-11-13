import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de SOP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .critical-field {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    .important-field {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
    }
    .reference-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .citation {
        font-size: 0.85rem;
        color: #555;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar modelo (simular - ajustar según tu modelo real)
@st.cache_resource
def load_model():
    try:
        modelo = joblib.load('C:/Codigos/Challenge/predictor-sop/modelos_pcos/modelo_gb_pcos.pkl')
        scaler = joblib.load('C:/Codigos/Challenge/predictor-sop/modelos_pcos/scaler_pcos.pkl')
        model_info = joblib.load('C:/Codigos/Challenge/predictor-sop/modelos_pcos/model_info.pkl')
        return modelo, scaler, model_info
    except:
        st.warning("Modelo no encontrado. Usando modo demo.")
        return None, None

# Función de predicción
def predecir_pcos(datos, modelo, scaler):
    """Realiza predicción de SOP"""

    # Crear DataFrame
    df_input = pd.DataFrame([datos])
    df_input = df_input[model_info['features']] # Reordenar columnas
        
    X_scaled = scaler.transform(df_input)
    probabilidad = modelo.predict_proba(X_scaled)[0][1]
    prediccion = "SOP" if probabilidad > 0.5 else "No SOP"
    
    if probabilidad < 0.3:
        nivel_riesgo = "🟢 Bajo"
        recomendacion = "Los indicadores sugieren bajo riesgo de SOP. Mantén controles periódicos."
    elif probabilidad < 0.7:
        nivel_riesgo = "🟡 Moderado"
        recomendacion = "Riesgo moderado. Se recomienda evaluación médica detallada."
    else:
        nivel_riesgo = "🔴 Alto"
        recomendacion = "Riesgo elevado. Consulta con un endocrinólogo o ginecólogo especialista."
    
    return {
        'prediccion': prediccion,
        'probabilidad': f"{probabilidad:.1%}",
        'probabilidad_num': probabilidad,
        'nivel_riesgo': nivel_riesgo,
        'recomendacion': recomendacion
    }

# Sidebar - Navegación
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["Predicción", "Referencias Médicas", "Información del Proyecto"]
)

modelo, scaler, model_info = load_model()

# ==================== PÁGINA 1: PREDICCIÓN ====================
if pagina == "Predicción":
    st.markdown('<h1 class="main-header">Predictor de Síndrome de Ovario Poliquístico (SOP)</h1>', 
                unsafe_allow_html=True)
    
    st.info("Complete los datos clínicos del paciente. Los campos marcados en rojo son los más importantes para la predicción.")
    
    with st.form("formulario_prediccion"):
        st.subheader("Datos mas importantes")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            follicle_r = st.number_input(
                "Folículos (Derecho)", 
                min_value=0, max_value=30, value=5,
                help="Número de folículos en ovario derecho. Normal: 2-10, SOP: >12"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            hair_growth = st.selectbox(
                "Crecimiento de vello (Hirsutismo)",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Crecimiento excesivo de vello en patrón masculino"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            weight_gain = st.selectbox(
                "Aumento de peso",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Ganancia de peso inexplicable o dificultad para perder peso"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            skin_darkening = st.selectbox(
                "Oscurecimiento de piel (Acantosis)",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Manchas oscuras en cuello, axilas o ingles"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            endometrium = st.number_input(
                "Grosor Endometrial (mm)",
                min_value=3.0, max_value=20.0, value=8.0, step=0.1,
                help="Grosor del endometrio medido por ecografía. Normal: 7-14mm"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            pimples = st.selectbox(
                "Acné",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Presencia de acné persistente"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Datos importantes")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            amh = st.number_input(
                "AMH (ng/mL)",
                min_value=0.0, max_value=20.0, value=3.0, step=0.1,
                help="Hormona Antimülleriana. Normal: <4.0, SOP: >4.7"
            )
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            
            cycle_length = st.number_input(
                "Días de sangrado menstrual",
                min_value=0, max_value=12, value=5,
                help="Duración del sangrado menstrual (0-12 días). Normal: 3-7 días"
            )
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            bmi = st.number_input(
                "IMC (kg/m²)",
                min_value=15.0, max_value=45.0, value=23.0, step=0.1,
                help="Índice de Masa Corporal"
            )
        
        with col5:
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            fsh_lh = st.number_input(
                "Ratio FSH/LH",
                min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                help="Relación FSH/LH. Normal: >2, SOP: <1"
            )
            
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            follicle_l = st.number_input(
                "Folículos (Izquierdo)",
                min_value=0, max_value=30, value=5,
                help="Número de folículos en ovario izquierdo"
            )
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            age = st.number_input(
                "Edad (años)",
                min_value=15, max_value=50, value=28,
                help="Edad del paciente"
            )
        
        with col6:
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            avg_f_size_l = st.number_input(
                "Tamaño promedio folículos (I) (mm)",
                min_value=2.0, max_value=25.0, value=10.0, step=0.1,
                help="Tamaño promedio de folículos ovario izquierdo"
            )
            
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            cycle_ri = st.selectbox(
                "Regularidad del ciclo",
                options=[2, 4],
                format_func=lambda x: "Regular" if x == 2 else "Irregular",
                help="Regularidad del ciclo menstrual"
            )
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            waist = st.number_input(
                "Circunferencia de cintura (cm)",
                min_value=50, max_value=150, value=80,
                help="Medida de la cintura"
            )
        
        st.markdown("---")
        st.subheader("Datos complementarios")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            fsh = st.number_input(
                "FSH (mIU/mL)",
                min_value=0.0, max_value=25.0, value=5.0, step=0.1,
                help="Hormona Folículo Estimulante"
            )
            
            pulse_rate = st.number_input(
                "Frecuencia cardíaca (bpm)",
                min_value=50, max_value=120, value=72,
                help="Pulso en reposo"
            )
        
        with col8:
            hb = st.number_input(
                "Hemoglobina (g/dL)",
                min_value=8.0, max_value=18.0, value=12.0, step=0.1,
                help="Nivel de hemoglobina"
            )
            
            hair_loss = st.selectbox(
                "Caída de cabello",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí"
            )
        
        with col9:
            fast_food = st.selectbox(
                "Consumo de comida rápida",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Consumo regular de comida rápida"
            )
            
            hip = st.number_input(
                "Circunferencia de cadera (cm)",
                min_value=70, max_value=150, value=100
            )
        
        # Botón de predicción
        submitted = st.form_submit_button("Realizar Predicción", type="primary", use_container_width=True)
        
        if submitted:
            # Preparar datos
            datos = {
                'Follicle_R': follicle_r,
                'Follicle_L': follicle_l,
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
            
            # Predecir
            resultado = predecir_pcos(datos, modelo, scaler)
            
            # Mostrar resultados
            st.markdown("---")
            st.markdown("## Resultados de la Predicción")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric(
                    "Predicción",
                    resultado['prediccion'],
                    delta=None,
                    delta_color="inverse"
                )
            
            with col_r2:
                st.metric(
                    "Probabilidad",
                    resultado['probabilidad']
                )
            
            with col_r3:
                st.metric(
                    "Nivel de Riesgo",
                    resultado['nivel_riesgo']
                )
            
            # Gráfico de probabilidad
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = resultado['probabilidad_num'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidad de SOP (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendación
            st.info(f"**Recomendación:** {resultado['recomendacion']}")
            
            st.warning("**Disclaimer:** Esta herramienta es de apoyo educativo y NO sustituye el diagnóstico médico profesional.")

# ==================== PÁGINA 2: REFERENCIAS MÉDICAS ====================
elif pagina == "Referencias Médicas":
    st.markdown('<h1 class="main-header">Referencias de Valores Normales y Literatura Científica</h1>', 
                unsafe_allow_html=True)
    
    st.info("Esta sección contiene rangos de referencia basados en literatura científica revisada por pares.")
    
    # Tabs para organizar referencias
    tab1, tab2, tab3, tab4 = st.tabs([
        "Biomarcadores Hormonales",
        "Variables Antropométricas",
        "Variables Clínicas",
        "Referencias Bibliográficas"
    ])
    
    with tab1:
        st.subheader("Biomarcadores Hormonales")
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### AMH (Hormona Antimülleriana)")
        st.markdown("""
        **Valores de Referencia:**
        - Normal: < 4.0 ng/mL
        - SOP: > 4.7 ng/mL (criterio diagnóstico 2024)
        - SOP severo: > 7.0 ng/mL
        
        **Importancia Clínica:**
        La AMH es el biomarcador más específico para SOP. Niveles elevados reflejan el exceso 
        de folículos antrales pequeños característico del síndrome.
        
        <p class="citation">Dewailly D, et al. (2014). "Diagnosis of polycystic ovary syndrome (PCOS): revisiting the threshold 
        values of follicle count on ultrasound and of the serum AMH level for the definition of polycystic ovaries." 
        Human Reproduction, 29(11): 2427-2436. DOI: 10.1093/humrep/deu234</p>
        
        <p class="citation">Teede HJ, et al. (2023). "Recommendations from the 2023 International Evidence-based Guideline 
        for the Assessment and Management of Polycystic Ovary Syndrome." Journal of Clinical Endocrinology & Metabolism, 
        108(10): 2447-2469. DOI: 10.1210/clinem/dgad463</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### Ratio FSH/LH")
        st.markdown("""
        **Valores de Referencia:**
        - Normal: > 2.0
        - SOP: < 1.0 (ratio invertido)
        - Fase folicular: 1.0 - 2.5
        
        **Importancia Clínica:**
        En SOP, la LH está elevada en relación a FSH, resultando en un ratio invertido. 
        Este desequilibrio contribuye a la anovulación y exceso de andrógenos.
        
        <p class="citation">Patel S. (2018). "Polycystic ovary syndrome (PCOS), an inflammatory, systemic, 
        lifestyle endocrinopathy." Journal of Steroid Biochemistry and Molecular Biology, 182: 27-36. 
        DOI: 10.1016/j.jsbmb.2018.04.008</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### FSH (Hormona Folículo Estimulante)")
        st.markdown("""
        **Valores de Referencia (Fase Folicular):**
        - Normal: 3.0 - 10.0 mIU/mL
        - SOP: Generalmente normal o ligeramente bajo
        - Menopausia: > 25 mIU/mL
        
        **Importancia Clínica:**
        En SOP, la FSH suele estar en rango normal-bajo. Se usa principalmente para calcular 
        el ratio FSH/LH y descartar otras causas de anovulación.
        
        <p class="citation">Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group (2004). 
        "Revised 2003 consensus on diagnostic criteria and long-term health risks related to polycystic ovary syndrome." 
        Fertility and Sterility, 81(1): 19-25. DOI: 10.1016/j.fertnstert.2003.10.004</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Variables Antropométricas")
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### IMC (Índice de Masa Corporal)")
        st.markdown("""
        **Clasificación OMS:**
        - Bajo peso: < 18.5 kg/m²
        - Normal: 18.5 - 24.9 kg/m²
        - Sobrepeso: 25.0 - 29.9 kg/m²
        - Obesidad Grado I: 30.0 - 34.9 kg/m²
        - Obesidad Grado II: 35.0 - 39.9 kg/m²
        - Obesidad Grado III: ≥ 40.0 kg/m²
        
        **Relación con SOP:**
        El 50-70% de mujeres con SOP tienen sobrepeso u obesidad. La obesidad exacerba 
        la resistencia a la insulina y el hiperandrogenismo.
        
        <p class="citation">Lim SS, et al. (2012). "Overweight, obesity and central obesity in women with polycystic 
        ovary syndrome: a systematic review and meta-analysis." Human Reproduction Update, 18(6): 618-637. 
        DOI: 10.1093/humupd/dms030</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### Circunferencia de Cintura")
        st.markdown("""
        **Valores de Referencia (Mujeres):**
        - Normal: < 80 cm (población asiática), < 88 cm (caucásica)
        - Riesgo aumentado: 80-88 cm
        - Riesgo muy alto: > 88 cm
        
        **Importancia Clínica:**
        Indicador de obesidad central y resistencia a la insulina, factores clave en SOP.
        
        <p class="citation">WHO (2008). "Waist Circumference and Waist-Hip Ratio: Report of a WHO Expert Consultation." 
        Geneva: World Health Organization.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Variables Clínicas")
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### Recuento de Folículos (FNPO)")
        st.markdown("""
        **Criterios de Rotterdam 2003 (Revisado):**
        - Normal: 2-10 folículos por ovario
        - SOP: ≥ 12 folículos (de 2-9 mm) por ovario
        - SOP (Criterio 2024): ≥ 20 folículos por ovario
        
        **Nota:** El criterio de 20 folículos es más específico con tecnología de ultrasonido moderna.
        
        <p class="citation">Dewailly D, et al. (2011). "Definition and significance of polycystic ovarian morphology: 
        a task force report from the Androgen Excess and Polycystic Ovary Syndrome Society." 
        Human Reproduction Update, 17(5): 667-685. DOI: 10.1093/humupd/dmr013</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### Días de Sangrado Menstrual")
        st.markdown("""
        **Valores Normales:**
        - Duración normal: 3-7 días
        - Amenorrea: 0 días (ausencia de menstruación)
        - Oligomenorrea: < 3 días
        - Menorragia: > 7 días
        
        **Relación con SOP:**
        La oligomenorrea y amenorrea son comunes en SOP debido a anovulación crónica.
        
        <p class="citation">Fraser IS, et al. (2011). "The FIGO recommendations on terminologies and definitions 
        for normal and abnormal uterine bleeding." Seminars in Reproductive Medicine, 29(5): 383-390. 
        DOI: 10.1055/s-0031-1287662</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### Grosor Endometrial")
        st.markdown("""
        **Valores de Referencia (Fase Folicular):**
        - Normal: 7-14 mm
        - Preocupante: > 14 mm (riesgo de hiperplasia)
        - SOP: Variable, mayor riesgo de hiperplasia endometrial
        
        **Importancia Clínica:**
        La anovulación crónica en SOP puede causar estimulación estrogénica prolongada 
        sin oposición de progesterona, aumentando el riesgo de hiperplasia endometrial.
        
        <p class="citation">Gallos ID, et al. (2012). "Regression, relapse, and live birth rates with fertility-preserving 
        therapy for endometrial hyperplasia and cancer." American Journal of Obstetrics & Gynecology, 207(4): 266.e1-12. 
        DOI: 10.1016/j.ajog.2012.08.011</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("### Hirsutismo (Escala de Ferriman-Gallwey)")
        st.markdown("""
        **Puntuación:**
        - Normal: < 8 puntos
        - Hirsutismo leve: 8-15 puntos
        - Hirsutismo moderado: 16-25 puntos
        - Hirsutismo severo: > 25 puntos
        
        **Prevalencia en SOP:**
        60-70% de mujeres con SOP presentan hirsutismo debido al exceso de andrógenos.
        
        <p class="citation">Yildiz BO, et al. (2010). "Prevalence, phenotype and cardiometabolic risk of polycystic 
        ovary syndrome under different diagnostic criteria." Human Reproduction, 25(5): 1229-1237. 
        DOI: 10.1093/humrep/deq020</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Referencias Bibliográficas Completas")
        
        st.markdown("""
        ### Guías Clínicas Internacionales
        
        1. **Teede HJ, et al. (2023).** "Recommendations from the 2023 International Evidence-based Guideline 
           for the Assessment and Management of Polycystic Ovary Syndrome." *Journal of Clinical Endocrinology & Metabolism*, 
           108(10): 2447-2469. DOI: 10.1210/clinem/dgad463
        
        2. **Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group (2004).** "Revised 2003 consensus on 
           diagnostic criteria and long-term health risks related to polycystic ovary syndrome." 
           *Fertility and Sterility*, 81(1): 19-25. DOI: 10.1016/j.fertnstert.2003.10.004
        
        ### Biomarcadores y Diagnóstico
        
        3. **Dewailly D, et al. (2014).** "Diagnosis of polycystic ovary syndrome (PCOS): revisiting the threshold 
           values of follicle count on ultrasound and of the serum AMH level for the definition of polycystic ovaries." 
           *Human Reproduction*, 29(11): 2427-2436. DOI: 10.1093/humrep/deu234
        
        4. **Dewailly D, et al. (2011).** "Definition and significance of polycystic ovarian morphology: a task force 
           report from the Androgen Excess and Polycystic Ovary Syndrome Society." *Human Reproduction Update*, 
           17(5): 667-685. DOI: 10.1093/humupd/dmr013
        
        ### Epidemiología y Factores de Riesgo
        
        5. **Lim SS, et al. (2012).** "Overweight, obesity and central obesity in women with polycystic ovary syndrome: 
           a systematic review and meta-analysis." *Human Reproduction Update*, 18(6): 618-637. 
           DOI: 10.1093/humupd/dms030
        
        6. **Yildiz BO, et al. (2010).** "Prevalence, phenotype and cardiometabolic risk of polycystic ovary syndrome 
           under different diagnostic criteria." *Human Reproduction*, 25(5): 1229-1237. DOI: 10.1093/humrep/deq020
        
        ### Fisiopatología
        
        7. **Patel S. (2018).** "Polycystic ovary syndrome (PCOS), an inflammatory, systemic, lifestyle endocrinopathy." 
           *Journal of Steroid Biochemistry and Molecular Biology*, 182: 27-36. DOI: 10.1016/j.jsbmb.2018.04.008
        
        ### Complicaciones y Manejo
        
        8. **Gallos ID, et al. (2012).** "Regression, relapse, and live birth rates with fertility-preserving therapy 
           for endometrial hyperplasia and cancer." *American Journal of Obstetrics & Gynecology*, 207(4): 266.e1-12. 
           DOI: 10.1016/j.ajog.2012.08.011
        
        9. **Fraser IS, et al. (2011).** "The FIGO recommendations on terminologies and definitions for normal and 
           abnormal uterine bleeding." *Seminars in Reproductive Medicine*, 29(5): 383-390. DOI: 10.1055/s-0031-1287662
        
        ### Organizaciones de Referencia
        
        10. **World Health Organization (WHO).** (2008). "Waist Circumference and Waist-Hip Ratio: Report of a WHO 
            Expert Consultation." Geneva: World Health Organization.
        
        ---
        
        ### Bases de Datos Consultadas
        - PubMed/MEDLINE
        - Cochrane Library
        - Google Scholar
        - Web of Science
        
        ### Última Actualización
        Noviembre 2025
        
        ### Nota Importante
        Esta información es para fines educativos. Los valores de referencia pueden variar según el laboratorio, 
        la población y el método de medición. Siempre consulte con un profesional de la salud calificado.
        """)

# ==================== PÁGINA 3: INFORMACIÓN DEL PROYECTO ====================
elif pagina == "Información del Proyecto":
    st.markdown('<h1 class="main-header">Información del Proyecto</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Análisis Estadístico de Biomarcadores y Factores Asociados al SOP
    
    ### Equipo: Los Poliquísticos
    - Bernardo Alejandro Partidas Díaz
    - Oscar Josue López González
    - Rodrigo Alonso Castillo Ramírez
    - Sebastian Sánchez Espinosa
    
    ### Institución
    Centro Universitario de Guadalajara (CUGDL) - Universidad de Guadalajara  
    **Asignatura:** Probabilidad y Estadística II  
    **Profesora:** Claudia Fabiola
    
    ### Dataset
    - **Total de observaciones:** 541 pacientes
    - **Total de variables:** 42 características
    - **Variables predictoras utilizadas:** 18
    
    ### Modelo de Machine Learning
    - **Algoritmo:** Gradient Boosting Classifier
    - **Accuracy:** 89.81%
    - **AUC-ROC:** 0.9542
    - **Variables más importantes:**
      1. Follicle_R (47.6%)
      2. Hair_growth (7.4%)
      3. Weight_gain (7.1%)
      4. Skin_darkening (5.9%)
      5. Endometrium (4.5%)
    
    ### Metodología Estadística
    - Pruebas de hipótesis (Mann-Whitney U, Chi-cuadrado)
    - Análisis de correlación (Spearman, V-Cramer)
    - Modelos de clasificación (Gradient Boosting, Random Forest, SVM, etc.)
    - Validación cruzada
    - Análisis de importancia de características (SHAP y sklearn)
    
    ### Objetivos del Proyecto
    1. Identificar patrones y diferencias significativas en variables clínicas, hormonales y de estilo de vida
    2. Desarrollar un modelo predictivo para SOP
    3. Determinar las variables más discriminantes
    4. Aplicar métodos de probabilidad y estadística en un caso biomédico real
    
    ### Resultados Esperados
    - Reporte estadístico completo
    - Modelo predictivo funcional
    - Visualizaciones interactivas
    - Aplicación web para uso clínico educativo
    
    ---
    
    ### Disclaimer
    Esta herramienta fue desarrollada con fines educativos como parte de un proyecto académico. 
    **NO debe utilizarse como herramienta de diagnóstico médico**. El diagnóstico de SOP debe 
    realizarse por un profesional de la salud calificado según los criterios de Rotterdam y las 
    guías clínicas internacionales.
    
    ### Contacto
    Para más información sobre este proyecto, contactar a través del Centro Universitario de Guadalajara (CUGDL).
    
    ---
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>Predictor de SOP v1.0</p>
    <p>Proyecto Académico - CUGDL UDG</p>
    <p>Noviembre 2025</p>
</div>
""", unsafe_allow_html=True)