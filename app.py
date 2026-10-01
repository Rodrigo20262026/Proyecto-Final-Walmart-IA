from __future__ import annotations

import streamlit as st

from modules.data import (
    load_default_data,
    load_uploaded_data,
    dataset_profile,
)

from modules.analytics import (
    missing_report,
    numeric_summary,
    target_correlations,
    train_regression_demo,
    build_agent_context,
)

from modules.visualizations import (
    histogram,
    scatter,
    correlation_heatmap,
    predicted_vs_real,
)

from modules.agent import LocalDataAgent, offline_summary


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Análisis de Ventas Walmart",
    page_icon="🤖",
    layout="wide",
)

st.title("Proyecto Final · Análisis de Ventas Walmart con Agente de IA")

st.caption(
    "Streamlit + Scikit-learn + Ollama · Proyecto Final Integrador"
)


# =========================================================
# BARRA LATERAL
# =========================================================

with st.sidebar:
    st.header("Configuración")

    uploaded = st.file_uploader(
        "CSV opcional",
        type=["csv"],
    )

    model_name = st.text_input(
        "Modelo Ollama",
        value="llama3.2:3b",
    )


# =========================================================
# CARGA DEL DATASET
# =========================================================

try:
    if uploaded is not None:
        df = load_uploaded_data(uploaded)
    else:
        df = load_default_data()

except Exception as exc:
    st.error(f"No fue posible cargar el dataset: {exc}")
    st.stop()


# =========================================================
# VARIABLES NUMÉRICAS
# =========================================================

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not numeric_cols:
    st.error("El proyecto requiere al menos una variable numérica.")
    st.stop()


# =========================================================
# VARIABLE OBJETIVO
# =========================================================

default_target = (
    "Weekly_Sales"
    if "Weekly_Sales" in numeric_cols
    else numeric_cols[-1]
)

target = st.sidebar.selectbox(
    "Variable objetivo",
    numeric_cols,
    index=numeric_cols.index(default_target),
)


# =========================================================
# PERFIL GENERAL DEL DATASET
# =========================================================

profile = dataset_profile(df)


# =========================================================
# MODELO DE REGRESIÓN
# =========================================================

model_result = None

try:
    model_result = train_regression_demo(df, target)
except Exception:
    model_result = None


# =========================================================
# AGENTE LOCAL OLLAMA
# =========================================================

agent = LocalDataAgent(model=model_name)

ollama_ok, installed_models, ollama_detail = agent.status()

with st.sidebar:
    st.markdown("---")

    if ollama_ok:
        st.success("Ollama disponible")

        if installed_models:
            st.caption(
                "Modelos instalados: "
                + ", ".join(installed_models[:5])
            )

    else:
        st.warning("Ollama no detectado")
        st.caption(
            "La aplicación continúa funcionando "
            "sin el agente generativo."
        )


# =========================================================
# INDICADORES PRINCIPALES
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Registros",
    profile["rows"],
)

c2.metric(
    "Variables",
    profile["columns"],
)

c3.metric(
    "Valores nulos",
    profile["missing"],
)

c4.metric(
    "Duplicados",
    profile["duplicates"],
)


# =========================================================
# PESTAÑAS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📄 Datos",
        "📊 Exploración",
        "📈 Modelo",
        "🤖 Agente",
        "🧩 Arquitectura",
    ]
)


# =========================================================
# PESTAÑA 1 - DATOS
# =========================================================

with tab1:

    st.subheader("Dataset de trabajo")

    if uploaded is None:
        st.info(
            "Dataset incluido: Walmart Sales. "
            "Contiene información de ventas semanales por tienda, "
            "fechas, feriados, temperatura, precio del combustible, "
            "índice de precios al consumidor (CPI) y desempleo. "
            "La variable objetivo principal es Weekly_Sales."
        )

    else:
        st.info(
            "Se está utilizando el archivo CSV "
            "cargado desde la interfaz."
        )

    st.markdown("#### Vista inicial del dataset")

    st.dataframe(
        df.head(25),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Calidad de datos")

    st.dataframe(
        missing_report(df),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Resumen numérico")

    st.dataframe(
        numeric_summary(df),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# PESTAÑA 2 - EXPLORACIÓN
# =========================================================

with tab2:

    st.subheader("Exploración visual")

    # Store es un identificador categórico, por lo que no se ofrece
    # como variable de distribución numérica.
    distribution_cols = [
        columna
        for columna in numeric_cols
        if columna != "Store"
    ]

    default_distribution = (
        "Weekly_Sales"
        if "Weekly_Sales" in distribution_cols
        else distribution_cols[0]
    )

    xcol = st.selectbox(
        "Variable para distribución",
        distribution_cols,
        index=distribution_cols.index(default_distribution),
    )

    st.plotly_chart(
        histogram(df, xcol),
        width="stretch",
    )

    candidates = [
        columna
        for columna in numeric_cols
        if columna != target
    ]

    if candidates:

        xscatter = st.selectbox(
            "Variable explicativa",
            candidates,
            index=0,
        )

        st.plotly_chart(
            scatter(
                df,
                xscatter,
                target,
            ),
            width="stretch",
        )

    st.markdown(
        "#### Correlaciones con la variable objetivo"
    )

    st.dataframe(
        target_correlations(
            df,
            target,
        ).head(10),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "#### Matriz de correlaciones"
    )

    st.plotly_chart(
        correlation_heatmap(df),
        width="stretch",
    )


# =========================================================
# PESTAÑA 3 - MODELO
# =========================================================

with tab3:

    st.subheader(
        "Modelo de regresión lineal"
    )

    st.write(
        f"Variable objetivo seleccionada: **{target}**"
    )

    if model_result is None:

        st.warning(
            "No fue posible entrenar el modelo "
            "con la configuración actual."
        )

    else:

        metrics = model_result["metrics"]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "R²",
            f"{metrics['r2']:.3f}",
        )

        m2.metric(
            "MAE",
            f"{metrics['mae']:.3f}",
        )

        m3.metric(
            "RMSE",
            f"{metrics['rmse']:.3f}",
        )

        st.caption(
            f"Entrenamiento: "
            f"{model_result['n_train']} registros · "
            f"Prueba: "
            f"{model_result['n_test']} registros"
        )

        st.markdown(
            "#### Valores reales vs. valores estimados"
        )

        st.plotly_chart(
            predicted_vs_real(
                model_result["predictions"]
            ),
            width="stretch",
        )

        st.markdown(
            "#### Coeficientes del modelo"
        )

        st.dataframe(
            model_result["coefficients"],
            width="stretch",
            hide_index=True,
        )


# =========================================================
# PESTAÑA 4 - AGENTE DE IA
# =========================================================

with tab4:

    st.subheader(
        "Agente local para interpretación"
    )

    context = build_agent_context(
        df,
        target,
        model_result,
    )

    st.caption(
        "El agente recibe un contexto estadístico "
        "resumido generado por Python. "
        "No recibe el dataset completo."
    )

    presets = [
        "Resume los principales hallazgos del dataset de Walmart.",
        "¿Qué variables presentan mayor relación con las ventas semanales?",
        "Interpreta las métricas del modelo de regresión.",
        "¿Existe alguna relación importante entre los feriados y las ventas?",
        "¿Qué factores podrían estar relacionados con las ventas semanales?",
        "¿Qué análisis adicional sería razonable realizar?",
    ]

    choice = st.selectbox(
        "Pregunta sugerida",
        ["Escribir otra pregunta..."]
        + presets,
    )

    question = st.text_area(
        "Pregunta",
        value=(
            ""
            if choice
            == "Escribir otra pregunta..."
            else choice
        ),
        height=100,
    )

    if st.button(
        "Analizar con el agente",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Escribe una pregunta."
            )

        elif ollama_ok:

            with st.spinner(
                "Consultando el modelo local..."
            ):

                try:

                    answer = agent.ask(
                        question.strip(),
                        context,
                    )

                    st.markdown(answer)

                except Exception as exc:

                    st.error(
                        "Ollama respondió "
                        f"con un error: {exc}"
                    )

                    st.markdown(
                        offline_summary(context)
                    )

        else:

            st.markdown(
                offline_summary(context)
            )

            st.caption(
                "Respuesta de respaldo: "
                "Ollama no está disponible "
                "en este equipo."
            )


# =========================================================
# PESTAÑA 5 - ARQUITECTURA
# =========================================================

with tab5:

    st.subheader(
        "Arquitectura modular del proyecto"
    )

    st.code(
        """
Walmart_Sales.csv
        ↓
modules/data.py
        ↓
modules/analytics.py
   ↙                 ↘
visualizations.py    contexto estadístico
   ↓                 ↓
Streamlit ← modules/agent.py → Ollama local
                              ↓
                         llama3.2:3b
""",
        language="text",
    )

    st.markdown(
        """
### Paradigmas utilizados

- **Programación imperativa:** controla la carga, validaciones, decisiones y flujo de la interfaz.
- **Programación funcional:** utiliza funciones reutilizables para perfilado, estadísticas, correlaciones, modelos y visualizaciones.
- **Programación orientada a objetos:** la clase `LocalDataAgent` encapsula la configuración y comunicación con Ollama.
- **Arquitectura modular:** las responsabilidades se encuentran separadas entre carga de datos, análisis, visualización, interfaz y agente de IA.

### Flujo del agente

1. Python procesa el dataset.
2. Se calculan estadísticas y resultados objetivos.
3. Se construye un contexto resumido.
4. El agente envía la pregunta y el contexto a Ollama.
5. `llama3.2:3b` genera una interpretación en lenguaje natural.
6. La respuesta se presenta dentro de Streamlit.
"""
    )


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.markdown("---")

st.caption(
    "Proyecto académico de análisis de ventas Walmart "
    "con Python, Streamlit, Scikit-learn y Ollama."
)
