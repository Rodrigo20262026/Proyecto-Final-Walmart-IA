# Arquitectura del Proyecto Final · Análisis de Ventas Walmart con Agente de IA

La aplicación utiliza una arquitectura modular que separa la carga de datos, el análisis estadístico, las visualizaciones, la interfaz y la comunicación con el modelo de inteligencia artificial local.

## Flujo general

```text
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
```

## Componentes principales

### app.py

Es el punto de entrada de la aplicación. Se encarga de:

* construir la interfaz con Streamlit;
* cargar los módulos del proyecto;
* mostrar tablas, métricas y gráficos;
* ejecutar el modelo de regresión;
* gestionar las preguntas al agente de inteligencia artificial.

### modules/data.py

Gestiona la carga y validación inicial de los datos.

Permite utilizar el archivo:

```text
data/Walmart_Sales.csv
```

También permite trabajar con un archivo CSV cargado desde la interfaz.

### modules/analytics.py

Contiene la lógica principal del análisis. Se encarga de:

* perfil general del dataset;
* identificación de valores nulos y duplicados;
* estadísticas descriptivas;
* análisis de correlaciones;
* comparación entre semanas con y sin feriado;
* análisis resumido por tienda;
* preparación de las variables;
* entrenamiento del modelo de regresión lineal;
* cálculo de R², MAE y RMSE;
* construcción del contexto estadístico utilizado por el agente.

La variable `Store` se trata como una variable categórica mediante variables dummy y no como una magnitud numérica continua.

### modules/visualizations.py

Contiene las funciones encargadas de generar los gráficos interactivos mediante Plotly.

Entre las visualizaciones se incluyen:

* distribución de ventas semanales;
* comparación de ventas según feriados;
* relaciones entre variables;
* matriz de correlaciones;
* valores reales frente a valores predichos.

### modules/agent.py

Contiene la clase:

```python
LocalDataAgent
```

Esta clase encapsula la comunicación con Ollama.

El agente recibe un contexto estadístico previamente calculado por Python y una pregunta realizada por el usuario.

El dataset completo no se envía directamente al modelo de lenguaje.

## Flujo del agente de IA

```text
Dataset
   ↓
Python procesa los datos
   ↓
Se calculan estadísticas y métricas
   ↓
Se construye un contexto resumido
   ↓
LocalDataAgent
   ↓
Ollama
   ↓
llama3.2:3b
   ↓
Interpretación en lenguaje natural
   ↓
Streamlit
```

Si Ollama no se encuentra disponible, la aplicación mantiene su funcionamiento analítico y presenta una respuesta descriptiva local de respaldo.

## Paradigmas de programación utilizados

### Programación imperativa

Controla el flujo de ejecución, validaciones, condiciones e interacción con el usuario.

### Programación funcional

Utiliza funciones reutilizables para:

* carga de datos;
* estadísticas;
* correlaciones;
* preparación del modelo;
* visualizaciones;
* construcción del contexto analítico.

### Programación orientada a objetos

Se aplica mediante la clase `LocalDataAgent`, que encapsula la configuración y comunicación con Ollama.

## Separación de responsabilidades

La arquitectura modular permite que cada archivo tenga una responsabilidad específica:

```text
data.py           → carga y preparación inicial de datos
analytics.py      → análisis estadístico y modelo
visualizations.py → gráficos interactivos
agent.py          → comunicación con Ollama
app.py            → interfaz y coordinación general
```

Esta separación facilita la reutilización del código, el mantenimiento del proyecto y la incorporación de nuevas funcionalidades.

