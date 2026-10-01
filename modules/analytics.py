from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


IDENTIFIER_COLUMNS = {"Store"}


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "variable": df.columns,
        "nulos": df.isna().sum().values,
        "porcentaje": (df.isna().mean().values * 100).round(2),
    })

    return (
        report
        .sort_values(["nulos", "variable"], ascending=[False, True])
        .reset_index(drop=True)
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").drop(
        columns=list(IDENTIFIER_COLUMNS),
        errors="ignore",
    )

    if numeric.empty:
        return pd.DataFrame()

    out = (
        numeric
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    return out.round(3)


def target_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").drop(
        columns=list(IDENTIFIER_COLUMNS),
        errors="ignore",
    )

    if target not in numeric.columns:
        return pd.DataFrame(
            columns=["variable", "correlacion"]
        )

    corr = (
        numeric
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
    )

    return (
        corr
        .rename("correlacion")
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
        .reset_index()
        .rename(columns={"index": "variable"})
    )


def _prepare_regression_data(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:

    if target not in df.columns:
        raise ValueError(
            "La variable objetivo no existe en el dataset."
        )

    work = df.copy()

    y = pd.to_numeric(
        work[target],
        errors="coerce",
    )

    numeric_predictors = [
        column
        for column in work.select_dtypes(
            include="number"
        ).columns
        if column not in {target, "Store"}
    ]

    parts: list[pd.DataFrame] = []

    if numeric_predictors:

        numeric_part = (
            work[numeric_predictors]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .astype(float)
        )

        parts.append(numeric_part)

    if "Store" in work.columns:

        store_dummies = pd.get_dummies(
            work["Store"],
            prefix="Store",
            drop_first=True,
            dtype=float,
        )

        parts.append(store_dummies)

    if not parts:
        raise ValueError(
            "No hay variables explicativas disponibles "
            "para entrenar el modelo."
        )

    X = pd.concat(
        parts,
        axis=1,
    )

    valid_rows = (
        y.notna()
        & X.notna().all(axis=1)
    )

    X = X.loc[
        valid_rows
    ].copy()

    y = y.loc[
        valid_rows
    ].copy()

    return X, y


def train_regression_demo(
    df: pd.DataFrame,
    target: str,
) -> dict:

    X, y = _prepare_regression_data(
        df,
        target,
    )

    if X.shape[1] == 0 or len(X) < 20:
        raise ValueError(
            "No hay suficientes variables o registros "
            "para entrenar el modelo."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    metrics = {
        "r2": float(
            r2_score(
                y_test,
                pred,
            )
        ),

        "mae": float(
            mean_absolute_error(
                y_test,
                pred,
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred,
                )
            )
        ),
    }

    coefficients = pd.DataFrame({
        "variable": X.columns,
        "coeficiente": model.coef_,
    }).sort_values(
        "coeficiente",
        key=lambda s: s.abs(),
        ascending=False,
    )

    predictions = pd.DataFrame({
        "real": y_test.to_numpy(),
        "predicho": pred,
    }).reset_index(
        drop=True
    )

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
        "features": list(X.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def _holiday_summary(
    df: pd.DataFrame,
    target: str,
) -> list[dict]:

    if (
        "Holiday_Flag" not in df.columns
        or target not in df.columns
    ):
        return []

    temp = df[
        ["Holiday_Flag", target]
    ].copy()

    temp[target] = pd.to_numeric(
        temp[target],
        errors="coerce",
    )

    temp["Holiday_Flag"] = pd.to_numeric(
        temp["Holiday_Flag"],
        errors="coerce",
    )

    temp = temp.dropna()

    if temp.empty:
        return []

    summary = (
        temp
        .groupby(
            "Holiday_Flag"
        )[target]
        .agg(
            ["count", "mean", "median"]
        )
        .reset_index()
    )

    summary["tipo_semana"] = (
        summary["Holiday_Flag"]
        .map({
            0: "Sin feriado",
            1: "Con feriado",
        })
        .fillna("Otro")
    )

    return (
        summary[
            [
                "tipo_semana",
                "count",
                "mean",
                "median",
            ]
        ]
        .round(3)
        .to_dict("records")
    )


def _store_summary(
    df: pd.DataFrame,
    target: str,
) -> dict:

    if (
        "Store" not in df.columns
        or target not in df.columns
    ):
        return {}

    temp = df[
        ["Store", target]
    ].copy()

    temp[target] = pd.to_numeric(
        temp[target],
        errors="coerce",
    )

    temp = temp.dropna()

    if temp.empty:
        return {}

    grouped = (
        temp
        .groupby(
            "Store"
        )[target]
        .agg(
            ["count", "mean", "median"]
        )
        .reset_index()
    )

    top_stores = (
        grouped
        .sort_values(
            "mean",
            ascending=False,
        )
        .head(5)
        .round(3)
        .to_dict("records")
    )

    return {
        "store_is_identifier": True,

        "number_of_stores": int(
            temp["Store"].nunique()
        ),

        "top_5_stores_by_average_sales":
            top_stores,
    }


def build_agent_context(
    df: pd.DataFrame,
    target: str,
    model_result: dict | None = None,
) -> dict:

    numeric = df.select_dtypes(
        include="number"
    )

    context = {
        "dataset": {
            "rows": int(
                df.shape[0]
            ),
            "columns": int(
                df.shape[1]
            ),
            "missing_total": int(
                df.isna().sum().sum()
            ),
            "duplicates": int(
                df.duplicated().sum()
            ),
            "target": target,
        },

        "variable_roles": {
            "Store": (
                "Identificador categórico de tienda. "
                "No debe interpretarse como una "
                "magnitud numérica continua."
            ),

            "Holiday_Flag": (
                "Variable binaria: "
                "0 = semana sin feriado, "
                "1 = semana con feriado."
            ),

            "Date": (
                "Variable temporal. "
                "No se utiliza directamente "
                "en el modelo lineal actual."
            ),
        },

        "target_summary": {},

        "top_correlations": [],

        "holiday_comparison":
            _holiday_summary(
                df,
                target,
            ),

        "store_summary":
            _store_summary(
                df,
                target,
            ),
    }

    if target in numeric.columns:

        s = pd.to_numeric(
            numeric[target],
            errors="coerce",
        ).dropna()

        context["target_summary"] = {
            "mean": round(
                float(s.mean()),
                3,
            ),

            "median": round(
                float(s.median()),
                3,
            ),

            "std": round(
                float(s.std()),
                3,
            ),

            "min": round(
                float(s.min()),
                3,
            ),

            "max": round(
                float(s.max()),
                3,
            ),
        }

        context["top_correlations"] = (
            target_correlations(
                df,
                target,
            )
            .head(5)
            .round(3)
            .to_dict("records")
        )

    if model_result:

        context["model"] = {
            "type":
                "LinearRegression",

            "r2":
                round(
                    model_result[
                        "metrics"
                    ]["r2"],
                    3,
                ),

            "mae":
                round(
                    model_result[
                        "metrics"
                    ]["mae"],
                    3,
                ),

            "rmse":
                round(
                    model_result[
                        "metrics"
                    ]["rmse"],
                    3,
                ),

            "n_train":
                model_result[
                    "n_train"
                ],

            "n_test":
                model_result[
                    "n_test"
                ],

            "store_encoding": (
                "Store fue tratada como "
                "variable categórica mediante "
                "variables dummy "
                "(one-hot encoding)."
            ),

            "holiday_flag": (
                "Holiday_Flag se incluyó "
                "como variable binaria 0/1."
            ),

            "date_usage": (
                "Date no se incluyó "
                "directamente en este "
                "modelo lineal."
            ),
        }

    return context