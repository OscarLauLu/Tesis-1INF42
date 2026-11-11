# Predicción de Tráfico y Velocidad de Flujo Vehicular en Lima Metropolitana mediante Deep Learning y Grafos Viales

**Curso:** [1INF42] Tesis 1  
**Institución:** Pontificia Universidad Católica del Perú (PUCP) - Facultad de Ciencias e Ingeniería  
**Autor:** Oscar Lau Lu  
**Asesor:** Dr. César Armando Beltrán Castañón  

---

## 📌 Descripción del Proyecto

Este repositorio contiene la base de código para el desarrollo de la investigación de tesis orientada al modelado y predicción espacio-temporal de flujo vehicular en Lima Metropolitana utilizando datos de vehículos flotantes (*Floating Car Data* - FCD de TomTom) y arquitecturas de Redes Neuronales sobre Grafos (GCN, STGCN, DCRNN).

---

## 📂 Estructura del Repositorio

La arquitectura del proyecto sigue las mejores prácticas de estructuración para proyectos de Ciencia de Datos y Deep Learning:

```text
Tesis-1INF42/
├── .gitignore                      # Reglas de exclusión para datasets, logs y temporales
├── README.md                       # Documentación general del repositorio
├── requirements.txt                # Lista de librerías y dependencias de Python
├── data/
│   ├── raw/                        # Datasets crudos de TomTom (no versionados en Git)
│   │   └── .gitkeep
│   └── processed/                  # Datasets preprocesados y tensores (T, N, F) (no versionados)
│       └── .gitkeep
├── notebooks/
│   └── 01_preprocesamiento_datos.ipynb  # Notebook interactivo de limpieza, EDA y regularización
├── src/
│   ├── __init__.py
│   └── preprocess_traffic.py       # Pipeline automatizado de preprocesamiento por CLI
└── results/
    ├── figures/                    # Gráficos y visualizaciones generadas
    │   └── grafico_patron_congestion.png
    ├── metrics/                    # Reportes estadísticos descriptivos (RE 2.2)
    │   └── traffic_summary_statistics.csv
    ├── models/                     # Parámetros serializados (MinMax scaler, pesos de modelos)
    │   └── scaler_params.json
    └── logs/                       # Trazabilidad y logs de ejecución (RE 2.1)
        └── preprocessing_execution.log
```

---

## 🔒 Política de Manejo de Datasets (Datos Sensibles / Licenciamiento)

> **IMPORTANTE:** Conforme a los Términos y Condiciones de uso de datos de TomTom y las buenas prácticas de Git, **ningún dataset (`.csv`, `.tsv`, `.npz`, `.parquet`) se sube al repositorio**.

1. Los directorios `data/raw/` y `data/processed/` están estrictamente ignorados en `.gitignore` (a excepción de sus archivos `.gitkeep`).
2. Para ejecutar localmente los experimentos, asegúrese de ubicar el dataset de telemetría original en:
   ```text
   data/raw/tomtom_lima_traffic_original.csv
   ```

---

## 🚀 Instalación y Preparación del Entorno

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/OscarLauLu/Tesis-1INF42.git
   cd Tesis-1INF42
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Linux/macOS
   # .venv\Scripts\activate   # En Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Ejecución del Pipeline de Preprocesamiento

### 1. Script por Línea de Comandos (`src/preprocess_traffic.py`)

El script ejecuta de forma automática:
- **RE 2.1:** Filtrado de anomalías ($v \in [0, 130]$ km/h, $v_{freeflow} > 0$), remoción de duplicados y regularización temporal (frecuencia de 15 min con interpolación lineal).
- **RE 2.2:** Generación del informe estadístico descriptivo por segmento vial.
- **RE 2.3:** Exportación del dataset limpio tabular y generación del tensor espacio-temporal multidimensional $(T, N, F)$ para Deep Learning.

```bash
# Ejecución básica con rutas predeterminadas (toma data/raw y escribe en data/processed y results)
python3 src/preprocess_traffic.py

# Ejecución con argumentos personalizados
python3 src/preprocess_traffic.py \
    --input data/raw/tomtom_lima_traffic_original.csv \
    --output data/processed/tomtom_traffic_preprocessed.csv \
    --tensor-output data/processed/traffic_tensor_34nodes.npz \
    --scaler-output results/models/scaler_params.json \
    --stats-output results/metrics/traffic_summary_statistics.csv \
    --log-file results/logs/preprocessing_execution.log \
    --freq 15min
```

### 2. Notebook Interactivo (`notebooks/01_preprocesamiento_datos.ipynb`)

Para realizar análisis exploratorio interactivo, visualización de series de tiempo y verificación paso a paso:

```bash
jupyter notebook notebooks/01_preprocesamiento_datos.ipynb
```

---

## 📊 Medios de Verificación Generados

- **`results/logs/preprocessing_execution.log`**: Registro cronológico de auditoría y transformaciones aplicadas.
- **`results/metrics/traffic_summary_statistics.csv`**: Métricas de dispersión y tendencia central por segmento vial.
- **`results/figures/grafico_patron_congestion.png`**: Curva horaria promedio de índice de congestión y velocidad.
- **`data/processed/traffic_tensor_34nodes.npz`**: Tensor de entrada tridimensional $(T, 34, 10)$ para los modelos de grafos.
