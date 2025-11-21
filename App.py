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
    .section-header {
        font-size: 1.8rem;
        color: #2c5282;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 10px;
        background-color: #e6f2ff;
        border-radius: 5px;
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
    .info-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar modelo
@st.cache_resource
def load_model():
    try:
        modelo = joblib.load('C:/Codigos/Challenge/predictor-sop/modelos_pcos/modelo_gb_pcos.pkl')
        scaler = joblib.load('C:/Codigos/Challenge/predictor-sop/modelos_pcos/scaler_pcos.pkl')
        model_info = joblib.load('C:/Codigos/Challenge/predictor-sop/modelos_pcos/model_info.pkl')
        return modelo, scaler, model_info
    except:
        st.warning("Modelo no encontrado. Usando modo demo.")
        return None, None, None

# Función de predicción
def predecir_pcos(datos, modelo, scaler):
    """Realiza predicción de SOP"""
    # Crear DataFrame
    df_input = pd.DataFrame([datos])
    df_input = df_input[model_info['features']]  # Reordenar columnas
        
    X = df_input
    probabilidad = modelo.predict_proba(X)[0][1]
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
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ¿Cómo funciona esta evaluación?
    
    Esta herramienta te ayudará a evaluar tu riesgo de SOP en dos pasos:
    
    1. **Síntomas Visibles**: Primero, responderás preguntas sobre síntomas que puedes identificar tú misma
    2. **Estudios Médicos** (opcional): Si tienes resultados de análisis o ultrasonidos, podrás agregarlos para una evaluación más precisa
    
    Si no cuentas con estudios médicos, ¡no te preocupes! La herramienta utilizará valores promedio para completar la evaluación.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("formulario_prediccion"):
        # ============== SECCIÓN 1: SÍNTOMAS VISIBLES ==============
        st.markdown('<div class="section-header">Síntomas que puedes identificar</div>', 
                    unsafe_allow_html=True)
        st.info("Responde estas preguntas basándote en lo que has observado en tu cuerpo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            weight_gain = st.selectbox(
                "¿Has experimentado aumento de peso sin razón aparente?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Ganancia de peso inexplicable o dificultad para perder peso"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            hair_growth = st.selectbox(
                "¿Tienes crecimiento excesivo de vello facial o corporal?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Vello en zonas como mentón, pecho, espalda, abdomen (patrón masculino)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            skin_darkening = st.selectbox(
                "¿Tienes manchas oscuras en la piel?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Manchas oscuras especialmente en cuello, axilas o ingles (acantosis nigricans)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            hair_loss = st.selectbox(
                "¿Has notado pérdida de cabello o calvicie?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Pérdida de cabello en patrón masculino (especialmente en la coronilla)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="critical-field">', unsafe_allow_html=True)
            pimples = st.selectbox(
                "¿Sufres de acné persistente?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
                help="Acné que no mejora con tratamientos convencionales"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            cycle_length = st.number_input(
                "¿Cuántos días dura tu sangrado menstrual?",
                min_value=0, max_value=12, value=5,
                help="Duración del sangrado menstrual. Normal: 3-7 días"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            cycle_ri = st.selectbox(
                "¿Tu ciclo menstrual es regular?",
                options=[2, 4],
                format_func=lambda x: "Regular (cada 21-35 días)" if x == 2 else "Irregular (varía mucho o ausente)",
                help="Regularidad del ciclo menstrual"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            fast_food = st.selectbox(
                "¿Consumes comida rápida frecuentemente?",
                options=[0, 1],
                format_func=lambda x: "No/Poco" if x == 0 else "Sí, frecuentemente",
                help="Consumo regular de comida rápida o procesada"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            age = st.number_input(
                "Edad",
                min_value=15, max_value=50, value=28,
                help="Tu edad actual"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="important-field">', unsafe_allow_html=True)
            bmi = st.number_input(
                "IMC (kg/m²)",
                min_value=15.0, max_value=45.0, value=23.0, step=0.1,
                help="Índice de Masa Corporal. Si no lo sabes, usa una calculadora online con tu peso y altura"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ============== SECCIÓN 2: ESTUDIOS MÉDICOS (OPCIONAL) ==============
        st.markdown("---")
        st.markdown('<div class="section-header">Estudios Médicos (Opcional)</div>', 
                    unsafe_allow_html=True)
        
        tiene_estudios = st.checkbox(
            "Tengo resultados de estudios médicos (análisis de sangre, ultrasonido, etc.)",
            value=False,
            help="Marca esta casilla si cuentas con resultados de laboratorio o ultrasonido"
        )
        
        if tiene_estudios:
            st.success("Excelente! Completa los estudios que tengas disponibles. Los campos que no conozcas se llenarán con valores normales.")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.markdown("#### Ultrasonido")
                follicle_r = st.number_input(
                    "Folículos (Derecho)",
                    min_value=0, max_value=30, value=5,
                    help="Número de folículos en ovario derecho. Normal: 2-10, SOP: >12"
                )
                
                endometrium = st.number_input(
                    "Grosor Endometrial (mm)",
                    min_value=3.0, max_value=20.0, value=8.0, step=0.1,
                    help="Grosor del endometrio medido por ecografía. Normal: 7-14mm"
                )
                
                avg_f_size_l = st.number_input(
                    "Tamaño promedio folículos (mm)",
                    min_value=2.0, max_value=25.0, value=10.0, step=0.1,
                    help="Tamaño promedio de folículos"
                )
            
            with col4:
                st.markdown("#### Hormonas")
                amh = st.number_input(
                    "AMH (ng/mL)",
                    min_value=0.0, max_value=20.0, value=3.0, step=0.1,
                    help="Hormona Antimülleriana. Normal: <4.0, SOP: >4.7"
                )
                
                fsh = st.number_input(
                    "FSH (mIU/mL)",
                    min_value=0.0, max_value=25.0, value=5.0, step=0.1,
                    help="Hormona Folículo Estimulante. Normal: 3-10 mIU/mL"
                )
                
                fsh_lh = st.number_input(
                    "Ratio FSH/LH",
                    min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                    help="Relación FSH/LH. Normal: >2, SOP: <1"
                )
            
            with col5:
                st.markdown("#### Otros")
                pulse_rate = st.number_input(
                    "Frecuencia cardíaca (bpm)",
                    min_value=50, max_value=120, value=72,
                    help="Pulso en reposo"
                )
                
                hb = st.number_input(
                    "Hemoglobina (g/dL)",
                    min_value=8.0, max_value=18.0, value=12.0, step=0.1,
                    help="Nivel de hemoglobina"
                )
        else:
            st.info("""
            No te preocupes! Usaremos valores normales promedio para los estudios médicos.
            
            La predicción se basará principalmente en los síntomas que has reportado, 
            que son muy importantes para la evaluación inicial del SOP.
            """)
            
            # Valores normales por defecto (SOLO LAS VARIABLES QUE EXISTEN EN EL MODELO ORIGINAL)
            follicle_r = 5
            endometrium = 8.0
            amh = 3.0
            fsh_lh = 1.0
            fsh = 5.0
            avg_f_size_l = 10.0
            pulse_rate = 72
            hb = 12.0
        
        # Botón de predicción
        st.markdown("---")
        submitted = st.form_submit_button("Realizar Predicción", use_container_width=True)
        
        if submitted:
            if modelo is None:
                st.error("El modelo no está disponible. Por favor, verifica la configuración.")
            else:
                # Preparar datos para predicción (SOLO LAS 18 VARIABLES DEL MODELO ORIGINAL)
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
                
                # Realizar predicción
                resultado = predecir_pcos(datos, modelo, scaler)
                
                # Mostrar resultados
                st.markdown("---")
                st.markdown("## Resultados de la Evaluación")
                
                # Métricas principales
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric(
                        "Diagnóstico Predicho",
                        resultado['prediccion'],
                        delta=None
                    )
                
                with col_res2:
                    st.metric(
                        "Probabilidad",
                        resultado['probabilidad'],
                        delta=None
                    )
                
                with col_res3:
                    st.metric(
                        "Nivel de Riesgo",
                        resultado['nivel_riesgo'],
                        delta=None
                    )
                
                # Gráfico de probabilidad
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=resultado['probabilidad_num'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidad de SOP (%)", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#90EE90'},
                            {'range': [30, 70], 'color': '#FFD700'},
                            {'range': [70, 100], 'color': '#FF6B6B'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recomendaciones
                st.markdown("### Recomendaciones")
                st.info(resultado['recomendacion'])
                
                # Información adicional basada en síntomas
                st.markdown("### Resumen de tus síntomas")
                
                sintomas_reportados = []
                if weight_gain == 1:
                    sintomas_reportados.append("Aumento de peso")
                if hair_growth == 1:
                    sintomas_reportados.append("Crecimiento excesivo de vello")
                if skin_darkening == 1:
                    sintomas_reportados.append("Manchas oscuras en la piel")
                if hair_loss == 1:
                    sintomas_reportados.append("Pérdida de cabello")
                if pimples == 1:
                    sintomas_reportados.append("Acné persistente")
                if cycle_ri == 4:
                    sintomas_reportados.append("Ciclo menstrual irregular")
                
                if sintomas_reportados:
                    st.write("Has reportado los siguientes síntomas:")
                    for sintoma in sintomas_reportados:
                        st.write(sintoma)
                else:
                    st.write("No has reportado síntomas significativos.")
                
                # Advertencia importante
                st.markdown("---")
                st.warning("""
                **IMPORTANTE**: Esta herramienta es solo para fines educativos y de orientación inicial.
                
                - NO reemplaza el diagnóstico médico profesional
                - El diagnóstico definitivo de SOP debe hacerlo un ginecólogo o endocrinólogo
                - Se requieren estudios médicos completos para confirmar el diagnóstico
                - Si tienes síntomas preocupantes, consulta a un profesional de la salud
                """)
                
                if not tiene_estudios:
                    st.info("""
                    📌 **Nota**: Esta predicción se realizó sin estudios médicos completos.
                    Para una evaluación más precisa, te recomendamos:
                    
                    1. Acudir con un ginecólogo o endocrinólogo
                    2. Solicitar un ultrasonido transvaginal
                    3. Realizar análisis hormonales (FSH, LH, AMH, etc.)
                    4. Volver a usar esta herramienta con tus resultados
                    """)

# ==================== PÁGINA 2: REFERENCIAS MÉDICAS ====================
elif pagina == "Referencias Médicas":
    st.markdown('<h1 class="main-header">Referencias Médicas y Valores de Referencia</h1>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Criterios Diagnósticos", 
        "Hormonas y Biomarcadores",
        "Valores Clínicos",
        "Bibliografía"
    ])
    
    with tab1:
        st.subheader("Criterios de Rotterdam para Diagnóstico de SOP")
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Criterios de Rotterdam (2003)
        
        El diagnóstico de SOP requiere **al menos 2 de los siguientes 3 criterios:**
        
        1. **Oligoovulación o anovulación**
           - Manifestada por irregularidades menstruales
           - Ciclos menstruales > 35 días o < 8 ciclos por año
        
        2. **Hiperandrogenismo clínico y/o bioquímico**
           - Clínico: Hirsutismo (escala Ferriman-Gallwey ≥ 8), acné, alopecia
           - Bioquímico: Niveles elevados de andrógenos (testosterona, DHEA-S, androstenediona)
        
        3. **Ovarios poliquísticos en ultrasonido**
           - ≥12 folículos de 2-9 mm de diámetro en al menos un ovario, o
           - Volumen ovárico aumentado (>10 cm³)
        
        **Importante:** Se deben excluir otras causas de hiperandrogenismo y anovulación.
        
        <p class="citation">Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group. (2004). 
        "Revised 2003 consensus on diagnostic criteria and long-term health risks related to polycystic ovary syndrome." 
        Fertility and Sterility, 81(1): 19-25. DOI: 10.1016/j.fertnstert.2003.10.004</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Guías Internacionales 2023
        
        Las guías más recientes enfatizan:
        - Uso de ultrasonido transvaginal de alta resolución
        - Umbral aumentado: ≥20 folículos por ovario (en mujeres adultas)
        - Consideración del contexto clínico completo
        - Evaluación de comorbilidades metabólicas
        
        <p class="citation">Teede HJ, et al. (2023). "Recommendations from the 2023 International Evidence-based 
        Guideline for the Assessment and Management of Polycystic Ovary Syndrome." 
        Journal of Clinical Endocrinology & Metabolism, 108(10): 2447-2469. DOI: 10.1210/clinem/dgad463</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Valores de Referencia - Hormonas y Biomarcadores")
        
        # Tabla de valores hormonales
        hormones_data = {
            'Hormona': ['FSH', 'LH', 'Relación LH/FSH', 'AMH', 'TSH', 'Prolactina', 'Vitamina D3'],
            'Unidad': ['mIU/mL', 'mIU/mL', 'Ratio', 'ng/mL', 'mIU/L', 'ng/mL', 'ng/mL'],
            'Rango Normal': ['3-10', '2-15', '<2', '<4.0', '0.4-4.0', '<25', '>30'],
            'SOP': ['Normal/bajo', 'Elevado', '>2-3', '>4.7', 'Normal', 'Normal/elevado', 'Frecuentemente bajo'],
            'Significado Clínico': [
                'Fase folicular. Bajo puede indicar función ovárica disminuida',
                'Elevado en SOP. Estimula producción de andrógenos',
                'Invertida en SOP. Indicador diagnóstico importante',
                'Marcador de reserva ovárica. Muy elevado en SOP',
                'Detecta disfunción tiroidea que puede simular SOP',
                'Elevada puede causar amenorrea. Excluir prolactinoma',
                'Deficiencia común en SOP. Afecta metabolismo'
            ]
        }
        
        df_hormones = pd.DataFrame(hormones_data)
        st.dataframe(df_hormones, use_container_width=True, hide_index=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Hormona Antimülleriana (AMH)
        
        **Valores de Referencia:**
        - Normal: 1.0 - 4.0 ng/mL
        - SOP: > 4.7 ng/mL (alta especificidad)
        - Reserva ovárica baja: < 1.0 ng/mL
        
        **Importancia en SOP:**
        La AMH es producida por folículos antrales pequeños. En SOP, el número excesivo de 
        folículos resulta en niveles muy elevados de AMH, convirtiéndola en un marcador 
        diagnóstico valioso con alta especificidad (88%) y sensibilidad (82%).
        
        <p class="citation">Dewailly D, et al. (2014). "Diagnosis of polycystic ovary syndrome (PCOS): 
        revisiting the threshold values of follicle count on ultrasound and of the serum AMH level 
        for the definition of polycystic ovaries." Human Reproduction, 29(11): 2427-2436. 
        DOI: 10.1093/humrep/deu234</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Relación LH/FSH
        
        **Interpretación:**
        - Normal: < 2
        - SOP clásico: 2-3 o mayor
        - Sensibilidad: ~40% (no todas las mujeres con SOP tienen relación elevada)
        
        **Fisiopatología:**
        La secreción aumentada de LH en SOP estimula las células de la teca ovárica para 
        producir más andrógenos, contribuyendo al hiperandrogenismo característico.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Valores Clínicos y Antropométricos")
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Índice de Masa Corporal (IMC/BMI)
        
        **Clasificación OMS:**
        - Bajo peso: < 18.5 kg/m²
        - Normal: 18.5 - 24.9 kg/m²
        - Sobrepeso: 25.0 - 29.9 kg/m²
        - Obesidad Grado I: 30.0 - 34.9 kg/m²
        - Obesidad Grado II: 35.0 - 39.9 kg/m²
        - Obesidad Grado III: ≥ 40.0 kg/m²
        
        **Relevancia en SOP:**
        - 40-80% de mujeres con SOP tienen sobrepeso/obesidad
        - La obesidad agrava la resistencia a la insulina y el hiperandrogenismo
        - Pérdida de 5-10% del peso puede mejorar significativamente los síntomas
        
        <p class="citation">Lim SS, et al. (2012). "Overweight, obesity and central obesity in women 
        with polycystic ovary syndrome: a systematic review and meta-analysis." Human Reproduction Update, 
        18(6): 618-637. DOI: 10.1093/humupd/dms030</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Relación Cintura-Cadera (Waist-Hip Ratio)
        
        **Valores de Riesgo (Mujeres):**
        - Normal: < 0.80
        - Riesgo moderado: 0.80 - 0.85
        - Riesgo alto: > 0.85
        
        **Circunferencia de Cintura:**
        - Normal: < 80 cm
        - Riesgo elevado: ≥ 80 cm
        - Riesgo muy elevado: ≥ 88 cm
        
        **Importancia:**
        La obesidad central (distribución androide) es común en SOP y se asocia con:
        - Mayor resistencia a la insulina
        - Riesgo cardiovascular aumentado
        - Perfil metabólico adverso
        
        <p class="citation">World Health Organization. (2008). "Waist Circumference and Waist-Hip Ratio: 
        Report of a WHO Expert Consultation." Geneva: World Health Organization.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Morfología Ovárica en Ultrasonido
        
        **Criterios de Rotterdam (Actualizados 2014):**
        - Folículos: ≥ 12 folículos de 2-9 mm por ovario (transductores antiguos)
        - Folículos: ≥ 20 folículos de 2-9 mm por ovario (transductores modernos ≥8 MHz)
        - Volumen ovárico: > 10 cm³
        
        **Técnica Recomendada:**
        - Ultrasonido transvaginal preferido sobre transabdominal
        - Evaluación en fase folicular temprana (días 3-5 del ciclo)
        - Medición 2D en corte transversal del ovario
        
        <p class="citation">Dewailly D, et al. (2011). "Definition and significance of polycystic ovarian morphology: 
        a task force report from the Androgen Excess and Polycystic Ovary Syndrome Society." 
        Human Reproduction Update, 17(5): 667-685. DOI: 10.1093/humupd/dmr013</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="reference-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Parámetros del Ciclo Menstrual
        
        **Duración del Ciclo:**
        - Normal: 21-35 días
        - Oligomenorrea: > 35 días
        - Amenorrea: Ausencia de menstruación por > 90 días
        
        **Duración del Sangrado:**
        - Normal: 3-7 días
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
      2. Hair_growth (7.5%)
      3. Weight_gain (7.2%)
      4. Skin_darkening (5.9%)
      5. AMH (4.4%)
    
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
    <p>Predictor de SOP v2.0</p>
    <p>Proyecto Académico - CUGDL UDG</p>
    <p>Noviembre 2025</p>
    <p><em>Versión Accesible</em></p>
</div>
""", unsafe_allow_html=True)