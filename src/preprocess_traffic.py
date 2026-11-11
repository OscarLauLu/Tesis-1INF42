#!/usr/bin/env python3
"""
================================================================================
SCRIPT DE PREPROCESAMIENTO DE DATOS DE TRÁFICO VEHICULAR (FCD - TOMTOM)
================================================================================
Proyecto de Tesis: Predicción de tráfico y velocidad de flujo vehicular en Lima
Metropolitana mediante técnicas de Deep Learning, empleando datos de vehículos
flotantes FCD y grafos viales.
Autor: Oscar Lau Lu
Asesor: Dr. César Armando Beltrán Castañón

Objetivo Específico 2 (C2):
- RE 2.1: Limpieza de datos (filtrado de anomalías, imputación y regularización)
- RE 2.2: Estudio estadístico de los datos recopilados
- RE 2.3: Dataset unificado para ciencia de datos y modelos espacio-temporales

Referencias Metodológicas:
- Kaya & Utku Kalay (2025): Filtrado de velocidades anómalas y normalización.
- Li et al. (2021): Tratamiento e imputación de valores faltantes en series de tráfico.
- Buroni et al. (2021): Codificación cíclica (seno/coseno) de variables temporales.
- Geng et al. (2026): Normalización y regularización de cuadrícula temporal.
================================================================================
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def setup_logger(log_file: str) -> logging.Logger:
    """Configura el logger para emitir registros a consola y a archivo de log."""
    logger = logging.getLogger("TrafficPreprocessor")
    logger.setLevel(logging.INFO)

    # Evitar duplicar handlers si ya existen
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo de log
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_raw_data(filepath: str, logger: logging.Logger) -> pd.DataFrame:
    """Carga y valida la estructura del dataset crudo de TomTom."""
    logger.info(f"Cargando dataset desde: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo especificado no existe: {filepath}")

    # Determinar delimitador (coma o tabulador)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(filepath, sep=sep)
    logger.info(f"Dimensiones iniciales del archivo: {df.shape[0]} filas x {df.shape[1]} columnas")

    expected_cols = [
        "timestamp",
        "segment_id",
        "segment_name",
        "zone",
        "latitude",
        "longitude",
        "current_speed",
        "freeflow_speed",
        "confidence",
        "congestion_index",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el archivo: {missing}")

    # Conversión a timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(
        f"Rango temporal crudo: {df['timestamp'].min()} hasta {df['timestamp'].max()} "
        f"({df['segment_id'].nunique()} segmentos únicos)"
    )

    return df


def filter_anomalies_and_duplicates(
    df: pd.DataFrame,
    min_speed: float,
    max_speed: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Filtra anomalías de velocidad e índices imposibles, y elimina duplicados
    siguiendo los criterios de Kaya & Utku Kalay (2025).
    """
    initial_rows = len(df)
    logger.info("=== FASE 1: Filtrado de anomalías y duplicados ===")

    # 1. Eliminación de duplicados exactos (timestamp, segment_id)
    dup_mask = df.duplicated(subset=["timestamp", "segment_id"], keep="first")
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        logger.warning(f"Se detectaron y eliminaron {n_dups} registros duplicados de (timestamp, segment_id).")
        df = df[~dup_mask].copy()
    else:
        logger.info("No se detectaron registros duplicados de (timestamp, segment_id).")

    # 2. Filtrado de velocidades anómalas (físicamente imposibles en red urbana)
    speed_anomaly_mask = (
        (df["current_speed"] < min_speed)
        | (df["current_speed"] > max_speed)
        | (df["freeflow_speed"] <= 0)
        | (df["freeflow_speed"] > max_speed)
    )
    n_speed_anomalies = int(speed_anomaly_mask.sum())
    if n_speed_anomalies > 0:
        logger.warning(
            f"Se detectaron {n_speed_anomalies} registros con velocidades fuera del rango válido "
            f"[{min_speed}, {max_speed}] km/h. Fueron descartados."
        )
        df = df[~speed_anomaly_mask].copy()
    else:
        logger.info(f"Todas las velocidades se encuentran dentro del rango válido [{min_speed}, {max_speed}] km/h.")

    # 3. Filtrado de índices fuera de rango estándar [0, 1]
    index_anomaly_mask = (
        (df["congestion_index"] < 0.0)
        | (df["congestion_index"] > 1.0)
        | (df["confidence"] < 0.0)
        | (df["confidence"] > 1.0)
    )
    n_index_anomalies = int(index_anomaly_mask.sum())
    if n_index_anomalies > 0:
        logger.warning(
            f"Se detectaron {n_index_anomalies} registros con congestion_index o confidence fuera de [0, 1]. Descartados."
        )
        df = df[~index_anomaly_mask].copy()
    else:
        logger.info("Todos los valores de congestion_index y confidence están en [0.0, 1.0].")

    logger.info(
        f"Filas tras filtrado de anomalías: {len(df)} (eliminadas: {initial_rows - len(df)})"
    )
    return df


def regularize_time_and_impute(
    df: pd.DataFrame,
    freq: str,
    filter_complete_days: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Alinea los timestamps a intervalos regulares (ej. 15 minutos) e imputa
    valores faltantes para construir una cuadrícula espacio-temporal completa,
    según Li et al. (2021) y Geng et al. (2026).
    """
    logger.info(f"=== FASE 2: Regularización de cuadrícula temporal e imputación ({freq}) ===")

    # Redondeo temporal a la cuadrícula fija (15min)
    df["timestamp_aligned"] = df["timestamp"].dt.round(freq)

    # Si hay colisiones tras el redondeo, promediar numéricos y tomar el primer metadato
    if df.duplicated(subset=["timestamp_aligned", "segment_id"]).any():
        logger.info("Promediando registros que coinciden en el mismo intervalo temporal alineado...")
        agg_rules = {
            "timestamp": "first",
            "segment_name": "first",
            "zone": "first",
            "latitude": "first",
            "longitude": "first",
            "current_speed": "mean",
            "freeflow_speed": "mean",
            "confidence": "mean",
            "congestion_index": "mean",
        }
        df = df.groupby(["timestamp_aligned", "segment_id"], as_index=False).agg(agg_rules)

    # Filtrar solo periodos donde todos los segmentos piloto (34) fueron monitoreados
    all_segments = sorted(df["segment_id"].unique())
    n_total_segments = len(all_segments)
    logger.info(f"Total de segmentos identificados en el grafo: {n_total_segments}")

    if filter_complete_days:
        daily_segment_counts = df.groupby(df["timestamp_aligned"].dt.date)["segment_id"].nunique()
        complete_dates = daily_segment_counts[daily_segment_counts == n_total_segments].index
        logger.info(
            f"Filtro de red completa activo: conservando {len(complete_dates)} días con los "
            f"{n_total_segments} segmentos activos (descartando pilotos parciales iniciales)."
        )
        df = df[df["timestamp_aligned"].dt.date.isin(complete_dates)].copy()

    # Construir mapa de metadatos estáticos por segmento (nombre, zona, lat, lon)
    meta_cols = ["segment_id", "segment_name", "zone", "latitude", "longitude"]
    segment_meta = df[meta_cols].drop_duplicates(subset=["segment_id"]).set_index("segment_id")

    # Generar producto cartesiano: (todos los timestamps únicos) x (todos los segmentos)
    unique_timestamps = sorted(df["timestamp_aligned"].unique())
    logger.info(
        f"Pasos temporales únicos a sincronizar: {len(unique_timestamps)} "
        f"x {n_total_segments} nodos = {len(unique_timestamps) * n_total_segments} celdas esperadas."
    )

    full_grid = pd.MultiIndex.from_product(
        [unique_timestamps, all_segments],
        names=["timestamp_aligned", "segment_id"],
    ).to_frame().reset_index(drop=True)

    # Unir con los datos observados
    merged = pd.merge(
        full_grid,
        df,
        on=["timestamp_aligned", "segment_id"],
        how="left",
    )

    missing_count = int(merged["current_speed"].isna().sum())
    logger.info(f"Valores faltantes a imputar en la cuadrícula: {missing_count} ({missing_count / len(merged):.2%})")

    # Restaurar metadatos estáticos para filas imputadas
    merged["segment_name"] = merged["segment_id"].map(segment_meta["segment_name"])
    merged["zone"] = merged["segment_id"].map(segment_meta["zone"])
    merged["latitude"] = merged["segment_id"].map(segment_meta["latitude"])
    merged["longitude"] = merged["segment_id"].map(segment_meta["longitude"])
    merged["timestamp"] = merged["timestamp_aligned"]

    # Imputación por segmento: interpolación temporal lineal, seguida de ffill y bfill
    merged = merged.sort_values(by=["segment_id", "timestamp_aligned"]).reset_index(drop=True)
    numeric_impute_cols = ["current_speed", "freeflow_speed", "congestion_index", "confidence"]

    for col in numeric_impute_cols:
        # Interpolación lineal temporal dentro de cada segmento
        merged[col] = (
            merged.groupby("segment_id")[col]
            .transform(lambda s: s.interpolate(method="linear").bfill().ffill())
        )
        # Si aún queda algún NaN aislado, imputar con la media general de la columna
        if merged[col].isna().any():
            col_mean = df[col].mean()
            merged[col] = merged[col].fillna(col_mean)

    logger.info("Imputación temporal completada con éxito. Valores nulos restantes: 0")
    return merged.sort_values(by=["timestamp_aligned", "segment_id"]).reset_index(drop=True)


def engineer_temporal_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Genera características temporales y cíclicas recomendadas en la literatura
    (Buroni et al., 2021) para modelos predictivos de tráfico.
    """
    logger.info("=== FASE 3: Ingeniería de características (Feature Engineering) ===")

    ts = df["timestamp_aligned"]

    # Descomposición de tiempo
    hour_float = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek  # 0: Lunes, 6: Domingo

    # Codificación cíclica (Seno y Coseno) para evitar discontinuidad de medianoche
    df["sin_hour"] = np.sin(2 * np.pi * hour_float / 24.0).round(6)
    df["cos_hour"] = np.cos(2 * np.pi * hour_float / 24.0).round(6)

    # Codificación cíclica para el día de la semana
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).round(6)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).round(6)

    # Indicador de fin de semana (1 si sábado o domingo, 0 laboral)
    df["is_weekend"] = dow.isin([5, 6]).astype(int)

    # Indicador de hora punta en Lima Metropolitana (Mañana: 07:00-09:30, Noche: 17:30-20:30)
    is_morning_rush = (hour_float >= 7.0) & (hour_float <= 9.5)
    is_evening_rush = (hour_float >= 17.5) & (hour_float <= 20.5)
    df["is_rush_hour"] = (is_morning_rush | is_evening_rush).astype(int)

    # Razón de velocidad actual vs flujo libre (medida relativa de fluidez)
    df["speed_ratio"] = (df["current_speed"] / df["freeflow_speed"]).clip(0.0, 2.0).round(4)

    logger.info(
        "Nuevas variables generadas: ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', "
        "'is_weekend', 'is_rush_hour', 'speed_ratio']"
    )
    return df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_output_path: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict]:
    """
    Normaliza variables numéricas continuas a escala [0, 1] mediante MinMax
    y guarda los parámetros para posterior desescalado al evaluar métricas (RMSE, MAE).
    """
    logger.info("=== FASE 4: Normalización MinMax de variables ===")
    scaler_params = {}

    for col in feature_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        denom = col_max - col_min if col_max != col_min else 1.0

        scaler_params[col] = {"min": col_min, "max": col_max}
        df[f"{col}_norm"] = ((df[col] - col_min) / denom).clip(0.0, 1.0).round(6)
        logger.info(f"Normalizada '{col}': min={col_min:.2f}, max={col_max:.2f}")

    if scaler_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(scaler_output_path)), exist_ok=True)
        with open(scaler_output_path, "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=4)
        logger.info(f"Parámetros del escalador exportados a: {scaler_output_path}")

    return df, scaler_params


def export_spatiotemporal_tensor(
    df: pd.DataFrame,
    features: list,
    tensor_output_path: str,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Reestructura el dataset en un tensor tridimensional (T, N, F)
    optimizado para modelos de Deep Learning en grafos (GCN + LSTM/GRU).
    Donde:
      T = Número de instantes de tiempo
      N = 34 nodos viales
      F = Características por nodo
    """
    logger.info("=== FASE 5: Construcción de tensor espacio-temporal (T, N, F) ===")

    timestamps = sorted(df["timestamp_aligned"].unique())
    segments = sorted(df["segment_id"].unique())

    T = len(timestamps)
    N = len(segments)
    F = len(features)

    # Ordenar estrictamente por tiempo y luego por segmento
    df_sorted = df.sort_values(by=["timestamp_aligned", "segment_id"]).reset_index(drop=True)

    # Verificar que el número de filas sea exactamente T * N
    expected_rows = T * N
    if len(df_sorted) != expected_rows:
        raise ValueError(f"Error de dimensiones: se esperaban {expected_rows} filas pero hay {len(df_sorted)}.")

    tensor = df_sorted[features].values.reshape((T, N, F))
    logger.info(f"Tensor espacio-temporal generado con dimensiones: (T={T}, N={N}, F={F})")

    if tensor_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(tensor_output_path)), exist_ok=True)
        np.savez_compressed(
            tensor_output_path,
            data=tensor,
            timestamps=np.array([str(t) for t in timestamps]),
            segments=np.array(segments),
            features=np.array(features),
        )
        logger.info(f"Tensor y metadatos exportados exitosamente a: {tensor_output_path}")

    return tensor


def generate_statistical_summary(
    df: pd.DataFrame,
    output_path: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Genera un reporte estadístico resumido para sustentar el RE 2.2."""
    logger.info("=== FASE 6: Generación de reporte estadístico (RE 2.2) ===")

    target_cols = ["current_speed", "freeflow_speed", "congestion_index"]
    stats_by_segment = df.groupby("segment_id")[target_cols].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        stats_by_segment.to_csv(output_path)
        logger.info(f"Resumen estadístico por segmento guardado en: {output_path}")

    return stats_by_segment


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_input = os.path.join(project_root, "data", "raw", "tomtom_lima_traffic_original.csv")
    if not os.path.exists(default_input):
        default_input = "tomtom_lima_traffic_original.csv"

    default_output = os.path.join(project_root, "data", "processed", "tomtom_traffic_preprocessed.csv")
    default_tensor = os.path.join(project_root, "data", "processed", "traffic_tensor_34nodes.npz")
    default_scaler = os.path.join(project_root, "results", "models", "scaler_params.json")
    default_stats = os.path.join(project_root, "results", "metrics", "traffic_summary_statistics.csv")
    default_log = os.path.join(project_root, "results", "logs", "preprocessing_execution.log")

    parser = argparse.ArgumentParser(
        description="Script de preprocesamiento de datos de tráfico vehicular FCD (TomTom) para modelos espacio-temporales."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=default_input,
        help="Ruta al archivo CSV de entrada con datos de tráfico.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Ruta para guardar el CSV procesado.",
    )
    parser.add_argument(
        "--tensor-output",
        "-t",
        type=str,
        default=default_tensor,
        help="Ruta para exportar el tensor comprimido (T, N, F) en formato .npz.",
    )
    parser.add_argument(
        "--scaler-output",
        "-s",
        type=str,
        default=default_scaler,
        help="Ruta para guardar los parámetros del escalador MinMax en JSON.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=default_stats,
        help="Ruta para exportar las estadísticas descriptivas por segmento.",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        default=default_log,
        help="Ruta para almacenar el log de ejecución (medio de verificación RE 2.1).",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="15min",
        help="Frecuencia de remuestreo temporal (ej. 15min, 10min, 30min).",
    )
    parser.add_argument(
        "--filter-complete-days",
        action="store_true",
        help="Filtrar únicamente días donde los 34 segmentos completos estuvieron activos.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=130.0,
        help="Velocidad máxima físicamente razonable en vías urbanas (km/h).",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=0.0,
        help="Velocidad mínima admisible (km/h).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)

    logger.info("Iniciando pipeline de preprocesamiento de telemetría vial...")
    logger.info(f"Parámetros: input={args.input}, output={args.output}, freq={args.freq}")

    # 1. Carga
    df = load_raw_data(args.input, logger)

    # 2. Filtrado de anomalías y duplicados
    df_clean = filter_anomalies_and_duplicates(
        df,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        logger=logger,
    )

    # 3. Regularización temporal e imputación de vacíos
    df_regular = regularize_time_and_impute(
        df_clean,
        freq=args.freq,
        filter_complete_days=args.filter_complete_days,
        logger=logger,
    )

    # 4. Feature Engineering (características cíclicas y temporales)
    df_featured = engineer_temporal_features(df_regular, logger)

    # 5. Normalización
    norm_cols = ["current_speed", "freeflow_speed", "congestion_index"]
    df_final, scaler_params = normalize_features(
        df_featured,
        feature_cols=norm_cols,
        scaler_output_path=args.scaler_output,
        logger=logger,
    )

    # 6. Reporte Estadístico (RE 2.2)
    generate_statistical_summary(df_final, args.stats_output, logger)

    # 7. Exportación del CSV procesado tabular
    cols_order = [
        "timestamp_aligned",
        "segment_id",
        "segment_name",
        "zone",
        "latitude",
        "longitude",
        "current_speed",
        "freeflow_speed",
        "congestion_index",
        "confidence",
        "current_speed_norm",
        "freeflow_speed_norm",
        "congestion_index_norm",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "is_weekend",
        "is_rush_hour",
        "speed_ratio",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_final[cols_order].to_csv(args.output, index=False)
    logger.info(f"Dataset preprocesado tabular guardado en: {args.output}")

    # 8. Exportación del tensor (T, N, F) para Deep Learning en Grafos
    tensor_features = [
        "current_speed_norm",
        "freeflow_speed_norm",
        "congestion_index_norm",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "is_weekend",
        "is_rush_hour",
        "speed_ratio",
    ]
    export_spatiotemporal_tensor(
        df_final,
        features=tensor_features,
        tensor_output_path=args.tensor_output,
        logger=logger,
    )

    logger.info("✓ Pipeline de preprocesamiento finalizado con éxito.")
    logger.info(f"Medios de verificación generados: {args.output}, {args.log_file}, {args.stats_output}")


if __name__ == "__main__":
    main()
