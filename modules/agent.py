from __future__ import annotations

import json
import requests


OLLAMA_URL = "http://localhost:11434"


class LocalDataAgent:
    """
    Agente local para interpretar resultados estadísticos
    del proyecto de análisis de ventas Walmart.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = OLLAMA_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def status(self) -> tuple[bool, list[str], str]:
        """
        Verifica si Ollama está disponible
        y obtiene los modelos instalados.
        """

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2,
            )

            response.raise_for_status()

            models = [
                model.get("name", "")
                for model in response.json().get("models", [])
            ]

            return True, models, "OK"

        except Exception as exc:
            return False, [], str(exc)

    def ask(
        self,
        question: str,
        context: dict,
    ) -> str:
        """
        Envía a Ollama la pregunta y un contexto
        estadístico previamente calculado por Python.
        """

        system = (
            "Eres un asistente académico de análisis de datos "
            "especializado en el dataset Walmart Sales. "

            "Responde únicamente con base en el contexto estadístico entregado. "

            "No inventes cifras, variables ni resultados. "

            "Distingue los hechos observados de las interpretaciones. "

            "Una correlación indica asociación y no demuestra causalidad. "

            "No afirmes que una diferencia es estadísticamente significativa "
            "si el contexto no incluye una prueba de significancia. "

            "La variable Store es un identificador categórico de tienda, "
            "no una magnitud numérica continua. "

            "Holiday_Flag indica si una semana tiene feriado o no. "

            "CPI significa índice de precios al consumidor. "

            "Cuando expliques R², indica qué proporción de la variación "
            "de Weekly_Sales explica el modelo. "

            "Responde en español natural, claro, breve y académico. "

            "Evita frases rebuscadas, repeticiones y palabras innecesarias. "

            "Utiliza como máximo aproximadamente 300 palabras. "

            "Termina siempre la respuesta con una oración completa."
        )

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": (
                        "CONTEXTO:\n"
                        f"{json.dumps(context, ensure_ascii=False)}"
                        "\n\n"
                        "PREGUNTA:\n"
                        f"{question}"
                    ),
                },
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": 500,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=180,
        )

        response.raise_for_status()

        return response.json()["message"]["content"].strip()


def offline_summary(context: dict) -> str:
    """
    Genera un resumen básico cuando
    Ollama no está disponible.
    """

    dataset = context.get("dataset", {})
    target = context.get("target_summary", {})
    model = context.get("model", {})

    text = (
        f"El dataset contiene "
        f"{dataset.get('rows', 0)} registros y "
        f"{dataset.get('columns', 0)} variables, "
        f"con {dataset.get('missing_total', 0)} valores nulos."
    )

    if target:
        text += (
            " La variable objetivo presenta "
            f"una media de {target.get('mean')} "
            f"y una mediana de {target.get('median')}."
        )

    if model:
        text += (
            " El modelo de regresión lineal obtuvo "
            f"R²={model.get('r2')}, "
            f"MAE={model.get('mae')} "
            f"y RMSE={model.get('rmse')}."
        )

    text += (
        " Esta salida es descriptiva y académica. "
        "Las asociaciones observadas no implican causalidad."
    )

    return text
