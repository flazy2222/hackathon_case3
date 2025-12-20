from __future__ import annotations

import io
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from fastapi import UploadFile

import plotly.express as px
import plotly.io as pio

from .config import PLOTS_DIR


def _read_uploaded_table(upload: UploadFile) -> pd.DataFrame:
    """
    Читает UploadFile как CSV или Excel (xlsx/xls).
    """
    content = upload.file.read()
    upload.file.seek(0)
    bio = io.BytesIO(content)
    filename = (upload.filename or "").lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(bio)
    else:
        return pd.read_csv(bio)


def _detect_label_columns(df: pd.DataFrame) -> Tuple[str | None, str | None]:
    """
    Ищем:
    - колонку с уровнем негатива (0/1/2),
    - колонку с типом обращения (0/1/2/3).

    В первую очередь учитываем конкретные имена из кейса:
    - ensemble_prediction
    - Вид обращения
    """
    num_df = df.select_dtypes(include=[np.number])
    sentiment_col: str | None = None
    topic_col: str | None = None

    if "ensemble_prediction" in num_df.columns:
        sentiment_col = "ensemble_prediction"
    if "Вид обращения" in num_df.columns:
        topic_col = "Вид обращения"

    if sentiment_col is not None and topic_col is not None:
        return sentiment_col, topic_col

    for col in num_df.columns:
        vals = num_df[col].dropna().unique()
        if len(vals) == 0:
            continue
        uniq = set(int(v) for v in vals)

        if sentiment_col is None and uniq.issubset({0, 1, 2}) and 1 <= len(uniq) <= 3:
            sentiment_col = col
            continue

        if topic_col is None and uniq.issubset({0, 1, 2, 3}) and len(uniq) >= 2 and col != sentiment_col:
            topic_col = col

    return sentiment_col, topic_col


def _build_sentiment_plot(df: pd.DataFrame, col: str) -> str:
    vc = (
        df[col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    vc.columns = ["sentiment", "count"]

    mapping = {
        0: "0 — нейтральное",
        1: "1 — эмоциональное, без агрессии",
        2: "2 — выраженная агрессия",
    }
    vc["sentiment_label"] = vc["sentiment"].map(mapping).fillna(vc["sentiment"].astype(str))

    fig = px.bar(
        vc,
        x="sentiment_label",
        y="count",
        title="Распределение обращений по уровням негатива",
        template="plotly_white",
        labels={"sentiment_label": "Класс", "count": "Количество обращений"},
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = PLOTS_DIR / "nlp_sentiment_dist.html"
    pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
    return f"/static/plots/{fname.name}"


def _build_topic_plot(df: pd.DataFrame, col: str) -> str:
    vc = (
        df[col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    vc.columns = ["topic", "count"]

    mapping = {
        0: "0 — просьба что-то сделать / нейтральное",
        1: "1 — пользователь что-то не понял / не нашёл",
        2: "2 — сообщение о проблеме",
        3: "3 — сообщение о проблеме",
    }
    vc["topic_label"] = vc["topic"].map(mapping).fillna(vc["topic"].astype(str))

    fig = px.bar(
        vc,
        x="topic_label",
        y="count",
        title="Распределение обращений по типам",
        template="plotly_white",
        labels={"topic_label": "Тип обращения", "count": "Количество обращений"},
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = PLOTS_DIR / "nlp_topic_dist.html"
    pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
    return f"/static/plots/{fname.name}"


def _build_sentiment_topic_heatmap(df: pd.DataFrame, sent_col: str, topic_col: str) -> str:
    pivot = df.pivot_table(
        index=sent_col,
        columns=topic_col,
        values=df.columns[0],
        aggfunc="count",
        fill_value=0,
    )

    pivot = pivot.sort_index().sort_index(axis=1)

    fig = px.imshow(
        pivot,
        labels=dict(
            x="Тип обращения (0/1/2/3)",
            y="Уровень негатива (0/1/2)",
            color="Количество обращений",
        ),
        title="Карта совместного распределения: уровень негатива × тип обращения",
        template="plotly_white",
        text_auto=True,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = PLOTS_DIR / "nlp_sentiment_topic_heatmap.html"
    pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
    return f"/static/plots/{fname.name}"


def process_nlp_file(upload: UploadFile) -> Tuple[Dict[str, Any], List[str]]:
    """
    Основной вход для FastAPI:
    1) читаем файл с обращениями;
    2) детектим колонки с предсказаниями моделей;
    3) строим графики;
    4) считаем агрегаты по классам;
    5) формируем словарь с описанием моделей и датасета.
    """
    df = _read_uploaded_table(upload)

    sentiment_col, topic_col = _detect_label_columns(df)
    plot_urls: List[str] = []

    if sentiment_col is not None:
        plot_urls.append(_build_sentiment_plot(df, sentiment_col))

    if topic_col is not None:
        plot_urls.append(_build_topic_plot(df, topic_col))

    if sentiment_col is not None and topic_col is not None:
        plot_urls.append(_build_sentiment_topic_heatmap(df, sentiment_col, topic_col))

    # --- агрегаты для KPI ---
    sentiment_dist: Dict[int, Dict[str, float]] | None = None
    topic_dist: Dict[int, Dict[str, float]] | None = None
    aggressive_share: float | None = None

    if sentiment_col is not None:
        vc = df[sentiment_col].value_counts().sort_index()
        total = vc.sum()
        sentiment_dist = {
            int(k): {
                "count": int(v),
                "share": float(v / total) if total > 0 else 0.0,
            }
            for k, v in vc.items()
        }

    if topic_col is not None:
        vc_t = df[topic_col].value_counts().sort_index()
        total_t = vc_t.sum()
        topic_dist = {
            int(k): {
                "count": int(v),
                "share": float(v / total_t) if total_t > 0 else 0.0,
            }
            for k, v in vc_t.items()
        }

    if sentiment_col is not None:
        mask_agg = df[sentiment_col] == 2
        if topic_col is not None:
            mask_problem = df[topic_col].isin([2, 3])
            mask_agg = mask_agg | mask_problem
        aggressive_share = float(mask_agg.mean()) if len(df) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "dataset_info": {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "sentiment_column": sentiment_col,
            "topic_column": topic_col,
        },
        "aggregates": {
            "sentiment_dist": sentiment_dist,
            "topic_dist": topic_dist,
            "aggressive_share": aggressive_share,
        },
        "sentiment_model": {
            "description": (
                "Модель оценки уровня негатива в обращении. "
                "Классы: 0 — нейтрально, 1 — эмоционально, без агрессии, 2 — агрессия."
            ),
            "classes": [0, 1, 2],
            "f1_macro_val": 0.58,
            "f1_macro_test": 0.58,
            "notes": "Датасет несбалансирован, модель лёгкая, работает на CPU.",
        },
        "topic_model": {
            "description": (
                "Модель классификации типа обращения. "
                "Класс 0 — просьба что-то сделать / нейтральное, "
                "1 — пользователь что-то не понял / не нашёл, "
                "2/3 — сообщение о проблеме."
            ),
            "classes": [0, 1, 2, 3],
            "f1_macro_val": 0.84,
            "f1_macro_test": 0.75,
            "notes": "Классы более сбалансированы, модель стабильнее и также работает на CPU.",
        },
    }

    return metrics, plot_urls
