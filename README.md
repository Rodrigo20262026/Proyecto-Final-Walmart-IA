# Proyecto Final · Análisis de Ventas Walmart con Agente de IA

Proyecto académico desarrollado para el módulo **Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos**.

La aplicación integra análisis de datos, visualizaciones interactivas, regresión lineal y un agente de inteligencia artificial local mediante **Ollama**.

---

## Descripción del proyecto

El proyecto analiza el dataset **Walmart Sales**, compuesto por información de ventas semanales de diferentes tiendas.

El archivo utilizado es:

```text
data/Walmart_Sales.csv
```

El dataset contiene **6.435 registros y 8 variables**:

- `Store`: identificador de la tienda.
- `Date`: fecha de la observación.
- `Weekly_Sales`: ventas semanales.
- `Holiday_Flag`: indica si la semana corresponde a un feriado.
- `Temperature`: temperatura registrada.
- `Fuel_Price`: precio del combustible.
- `CPI`: índice de precios al consumidor.
- `Unemployment`: tasa de desempleo.

La variable objetivo principal del proyecto es:

```text
Weekly_Sales
```

---

## Objetivo

Desarrollar una aplicación modular en Python que permita:

- cargar y explorar un dataset;
- evaluar la calidad de los datos;
- calcular estadísticas descriptivas;
- analizar correlaciones;
- generar visualizaciones interactivas;
- entrenar un modelo de regresión lineal;
- interpretar resultados mediante un agente de IA local;
- aplicar programación imperativa, funcional y orientada a objetos.

---

## Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Statsmodels
- Requests
- Ollama
- Llama 3.2 3B

---

## Arquitectura modular

```text
Proyecto_Walmart/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── ARCHITECTURE.md
│
├── data/
│   └── Walmart_Sales.csv
│
└── modules/
    ├── __init__.py
    ├── data.py
    ├── analytics.py
    ├── visualizations.py
    └── agent.py
```

### Responsabilidad de cada módulo

**app.py**

Contiene la interfaz principal desarrollada con Streamlit y coordina los diferentes componentes de la aplicación.

**modules/data.py**

Gestiona la carga del dataset y el perfil general de los datos.

**modules/analytics.py**

Contiene las funciones de análisis estadístico, correlaciones, preparación de datos y modelo de regresión.

**modules/visualizations.py**

Genera las visualizaciones interactivas mediante Plotly.

**modules/agent.py**

Contiene la clase `LocalDataAgent`, encargada de comunicarse con Ollama y generar interpretaciones en lenguaje natural a partir del contexto estadístico calculado por Python.

---

## Tratamiento de las variables

La variable `Store` representa un identificador de tienda. Aunque está almacenada numéricamente, no se interpreta como una magnitud continua. Para el modelo de regresión se trata como una variable categórica mediante variables dummy.

`Holiday_Flag` se interpreta como una variable binaria:

```text
0 = semana sin feriado
1 = semana con feriado
```

La variable `Date` se conserva como información temporal, pero no se utiliza directamente como predictor en el modelo lineal actual.

---

## Análisis exploratorio

La aplicación permite realizar:

- perfil general del dataset;
- identificación de valores nulos;
- detección de registros duplicados;
- estadísticas descriptivas;
- distribución de `Weekly_Sales`;
- comparación de ventas entre semanas con y sin feriado;
- análisis de correlaciones;
- matriz de correlaciones.

El dataset utilizado presenta:

```text
Registros: 6435
Variables: 8
Valores nulos: 0
Duplicados: 0
```

---

## Modelo de regresión

Se utiliza un modelo de **regresión lineal** mediante Scikit-learn.

La variable objetivo es:

```text
Weekly_Sales
```

La variable `Store` se representa mediante variables indicadoras o dummy.

La división de los datos utilizada es:

```text
75 % entrenamiento
25 % prueba
```

Con la configuración actual se obtuvieron aproximadamente los siguientes resultados:

```text
R²   = 0.916
MAE  = 93,320.307
RMSE = 164,236.272
```

El R² indica que el modelo explica aproximadamente el **91,6 % de la variación observada en las ventas semanales dentro de los datos analizados**.

El R² no debe interpretarse como un porcentaje de precisión predictiva.

---

## Agente de inteligencia artificial local

El proyecto incorpora un agente local mediante **Ollama**.

Modelo utilizado:

```text
llama3.2:3b
```

El agente no recibe directamente todo el dataset. Python calcula previamente un contexto resumido que contiene información como:

- dimensiones del dataset;
- estadísticas de `Weekly_Sales`;
- correlaciones;
- comparación de semanas con y sin feriado;
- información resumida por tienda;
- métricas del modelo de regresión.

Posteriormente, ese contexto se envía al modelo local para generar una interpretación en lenguaje natural.

El agente está diseñado para utilizar los resultados calculados previamente por Python y no sustituye el análisis estadístico.

---

## Paradigmas de programación utilizados

### Programación imperativa

Se utiliza para controlar el flujo general de la aplicación, las validaciones, las decisiones y la interacción con el usuario.

### Programación funcional

Se utilizan funciones reutilizables para carga de datos, perfilado, estadísticas, correlaciones, preparación del modelo y visualizaciones.

### Programación orientada a objetos

Se utiliza mediante la clase:

```python
LocalDataAgent
```

Esta clase encapsula la configuración y comunicación con Ollama.

---

## Instalación

### 1. Crear un entorno virtual

En Windows:

```bash
python -m venv .venv
```

Si se utiliza Python 3.13 con `virtualenv`:

```bash
py -3.13 -m virtualenv .venv
```

Activar el entorno:

```bash
.venv\Scripts\activate
```

### 2. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### 3. Instalar Ollama

Instalar Ollama desde su sitio oficial y descargar el modelo:

```bash
ollama pull llama3.2:3b
```

Verificar los modelos instalados:

```bash
ollama list
```

### 4. Ejecutar la aplicación

Con el entorno virtual activado:

```bash
python -m streamlit run app.py
```

La aplicación estará disponible normalmente en:

```text
http://localhost:8501
```

---

## Funcionamiento general

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

---

## Consideraciones académicas

Las correlaciones presentadas en el proyecto representan asociaciones estadísticas y no implican causalidad.

Las comparaciones entre semanas con feriado y sin feriado son descriptivas y no deben interpretarse como pruebas de significancia estadística.

El agente de IA local genera interpretaciones a partir del contexto estadístico calculado por Python y puede presentar limitaciones propias del modelo de lenguaje utilizado.

---

## Ejecución sin Ollama

La parte analítica de la aplicación puede ejecutarse aunque Ollama no esté disponible.

En ese caso, el programa genera una respuesta descriptiva básica de respaldo.

---

## Autor

Proyecto desarrollado como trabajo académico para el módulo:

**Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos**
