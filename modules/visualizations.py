from __future__ import annotations

import pandas as pd
import plotly.express as px


IDENTIFIER_COLUMNS = {"Store"}


def histogram(df: pd.DataFrame, column: str):
    """
    Genera una visualización de distribución adecuada para la variable elegida.

    - Store se trata como identificador categórico.
    - Holiday_Flag se trata como variable binaria categórica.
    - Las demás variables numéricas se muestran con histograma y boxplot marginal.
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el dataset.")

    if column == "Store":
        counts = (
            df["Store"]
            .dropna()
            .value_counts()
            .sort_index()
        )

        chart_data = pd.DataFrame(
            {
                "Store": counts.index.astype(str),
                "Frecuencia": counts.values,
            }
        )

        fig = px.bar(
            chart_data,
            x="Store",
            y="Frecuencia",
            title="Frecuencia de registros por tienda",
            labels={
                "Store": "Tienda",
                "Frecuencia": "Número de registros",
            },
        )

        fig.update_layout(
            xaxis_type="category",
        )

        return fig

    if column == "Holiday_Flag":
        counts = (
            df["Holiday_Flag"]
            .dropna()
            .value_counts()
            .sort_index()
        )

        labels_map = {
            0: "Sin feriado",
            1: "Con feriado",
        }

        chart_data = pd.DataFrame(
            {
                "Tipo de semana": [
                    labels_map.get(value, str(value))
                    for value in counts.index
                ],
                "Frecuencia": counts.values,
            }
        )

        return px.bar(
            chart_data,
            x="Tipo de semana",
            y="Frecuencia",
            title="Distribución de Holiday_Flag",
            labels={
                "Tipo de semana": "Tipo de semana",
                "Frecuencia": "Número de registros",
            },
        )

    return px.histogram(
        df,
        x=column,
        marginal="box",
        title=f"Distribución de {column}",
        labels={
            column: column,
            "count": "Frecuencia",
        },
    )


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
):
    """
    Genera una visualización entre una variable explicativa y la variable objetivo.

    - Store se representa como categoría mediante boxplots por tienda.
    - Holiday_Flag se representa como comparación entre semanas con y sin feriado.
    - Para variables numéricas continuas se utiliza dispersión con tendencia lineal.
    """
    if x not in df.columns:
        raise ValueError(f"La columna '{x}' no existe en el dataset.")

    if y not in df.columns:
        raise ValueError(f"La columna '{y}' no existe en el dataset.")

    if x == "Store":
        chart_data = df[[x, y]].dropna().copy()
        chart_data["Store"] = chart_data["Store"].astype(int).astype(str)

        fig = px.box(
            chart_data,
            x="Store",
            y=y,
            points=False,
            title=f"{y} por tienda",
            labels={
                "Store": "Tienda",
                y: y,
            },
        )

        fig.update_layout(
            xaxis_type="category",
        )

        return fig

    if x == "Holiday_Flag":
        chart_data = df[[x, y]].dropna().copy()

        chart_data["Tipo de semana"] = chart_data[
            "Holiday_Flag"
        ].map(
            {
                0: "Sin feriado",
                1: "Con feriado",
            }
        ).fillna(
            chart_data["Holiday_Flag"].astype(str)
        )

        return px.box(
            chart_data,
            x="Tipo de semana",
            y=y,
            points=False,
            title=f"{y} según Holiday_Flag",
            labels={
                "Tipo de semana": "Tipo de semana",
                y: y,
            },
        )

    chart_data = df[[x, y]].dropna()

    return px.scatter(
        chart_data,
        x=x,
        y=y,
        trendline="ols",
        title=f"{y} vs. {x}",
        labels={
            x: x,
            y: y,
        },
    )


def correlation_heatmap(df: pd.DataFrame):
    """
    Genera la matriz de correlaciones únicamente con variables numéricas
    que representan magnitudes. Store se excluye porque es un identificador.
    """
    numeric_df = df.select_dtypes(include="number").copy()

    columns_to_drop = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in numeric_df.columns
    ]

    if columns_to_drop:
        numeric_df = numeric_df.drop(
            columns=columns_to_drop
        )

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "Se requieren al menos dos variables numéricas "
            "para construir la matriz de correlaciones."
        )

    corr = numeric_df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title="Matriz de correlaciones",
        labels={
            "x": "Variable",
            "y": "Variable",
            "color": "Correlación",
        },
    )

    return fig


def predicted_vs_real(predictions: pd.DataFrame):
    """
    Grafica valores reales frente a valores estimados.

    Acepta distintos nombres habituales de columnas para mantener
    compatibilidad con el resultado producido por analytics.py.
    """
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    candidate_pairs = [
        ("real", "predicted"),
        ("real", "predicho"),
        ("Real", "Predicho"),
        ("actual", "predicted"),
        ("Actual", "Predicted"),
        ("y_true", "y_pred"),
        ("observed", "predicted"),
    ]

    x_col = None
    y_col = None

    for left, right in candidate_pairs:
        if left in predictions.columns and right in predictions.columns:
            x_col = left
            y_col = right
            break

    if x_col is None or y_col is None:
        numeric_cols = predictions.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_cols) < 2:
            raise ValueError(
                "No fue posible identificar las columnas "
                "de valores reales y estimados."
            )

        x_col, y_col = numeric_cols[:2]

    chart_data = predictions[
        [x_col, y_col]
    ].dropna()

    fig = px.scatter(
        chart_data,
        x=x_col,
        y=y_col,
        title="Valores reales vs. valores estimados",
        labels={
            x_col: "Valor real",
            y_col: "Valor estimado",
        },
    )

    if not chart_data.empty:
        min_value = min(
            chart_data[x_col].min(),
            chart_data[y_col].min(),
        )
        max_value = max(
            chart_data[x_col].max(),
            chart_data[y_col].max(),
        )

        fig.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line={
                "dash": "dash",
            },
        )

    return fig
