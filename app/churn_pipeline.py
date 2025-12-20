from __future__ import annotations

import io
import time
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from fastapi import UploadFile

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import plotly.express as px
import plotly.io as pio

from .config import PLOTS_DIR


# ==============================
# Чтение загруженных файлов
# ==============================

def _read_uploaded_table(upload: UploadFile) -> pd.DataFrame:
    """
    Читает UploadFile как CSV или Excel синхронно.
    """
    content = upload.file.read()
    upload.file.seek(0)
    bio = io.BytesIO(content)
    filename = (upload.filename or "").lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(bio)
    else:
        return pd.read_csv(bio)


# ==============================
# Предобработка (как в ноутбуке)
# ==============================

def preprocess_data(events: pd.DataFrame, socdem: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая предобработка из ноутбука:
    - 'Дата и время события' -> datetime, Europe/Moscow, без tz
    - merge с соцдемом.
    """
    df = events.copy()
    col_dt = "Дата и время события"

    if col_dt not in df.columns:
        raise ValueError("Ожидается колонка 'Дата и время события' в логе событий")

    df[col_dt] = df[col_dt].astype(str).str.replace(r"\[.*\]$", "", regex=True)
    df[col_dt] = pd.to_datetime(df[col_dt], format="mixed", utc=True)
    df[col_dt] = df[col_dt].dt.tz_convert("Europe/Moscow")
    df[col_dt] = df[col_dt].dt.tz_localize(None)

    if "Идентификатор устройства" not in df.columns:
        raise ValueError("Ожидается колонка 'Идентификатор устройства'")

    if "number" not in socdem.columns:
        raise ValueError("В словаре соцдема ожидается колонка 'number'")

    merged = df.merge(
        socdem,
        how="left",
        left_on="Идентификатор устройства",
        right_on="number",
    ).drop(columns=["number"])

    return merged


# ==============================
# Построение final_df (как в ноутбуке)
# ==============================

def build_user_features(data: pd.DataFrame) -> pd.DataFrame:
    df_all = data.copy()
    col_dt = "Дата и время события"

    if col_dt not in df_all.columns:
        raise ValueError("В данных нет колонки 'Дата и время события'")

    df_all[col_dt] = pd.to_datetime(df_all[col_dt])

    # 0) dop_data — с ненулевым 'Действие'
    if "Действие" not in df_all.columns:
        raise ValueError("В данных нет колонки 'Действие'")

    dop_data = df_all.dropna(subset=["Действие"])

    # 1) churn по последнему событию
    ref_date = df_all[col_dt].max()

    last_activity = (
        df_all
        .groupby("Идентификатор устройства")[col_dt]
        .max()
        .reset_index()
        .rename(columns={col_dt: "last_event_time"})
    )

    last_activity["days_since_last"] = (
        ref_date - last_activity["last_event_time"]
    ).dt.days
    last_activity["churn"] = (last_activity["days_since_last"] >= 30).astype(int)

    # 2) фичи по активности и времени суток
    df = dop_data.copy()
    df[col_dt] = pd.to_datetime(df[col_dt])
    df["hour"] = df[col_dt].dt.hour

    def part_of_day(h: int) -> str:
        if 6 <= h < 12:
            return "утро"
        elif 12 <= h < 18:
            return "день"
        elif 18 <= h < 24:
            return "вечер"
        else:
            return "ночь"

    df["part_of_day"] = df["hour"].apply(part_of_day)

    g = df.groupby("Идентификатор устройства")

    agg = g.agg(
        last_age=("age_back", "last"),
        last_gender=("gender", "last"),
        actions_total=(col_dt, "count"),
        first_dt=(col_dt, "min"),
        last_dt=(col_dt, "max"),
        last_manufacturer=("Производитель устройства", "last"),
    )

    days = (agg["last_dt"] - agg["first_dt"]).dt.days.clip(lower=1)
    agg["actions_per_day"] = agg["actions_total"] / days

    part_share = (
        df.groupby(["Идентификатор устройства", "part_of_day"])
        .size()
        .unstack(fill_value=0)
    )
    part_share = part_share.div(part_share.sum(axis=1), axis=0)
    dominant_part = part_share.idxmax(axis=1).rename("main_part_of_day")

    final_df = agg.join(dominant_part)

    # 3) биннинг возраста, кодировка пола и части суток
    def part_of_day_e(x: str) -> int:
        if x == "утро":
            return 1
        if x == "день":
            return 2
        if x == "вечер":
            return 3
        if x == "ночь":
            return 4
        return 0

    bins = [0, 18, 25, 35, 45, 60, np.inf]
    labels = [1, 2, 3, 4, 5, 6]

    age_mean = final_df["last_age"].mean()
    gender_mode = final_df["last_gender"].mode()[0]

    final_df["last_age"] = pd.cut(
        final_df["last_age"].fillna(age_mean),
        bins=bins,
        labels=labels,
        right=False,
    ).astype(int)

    final_df["last_gender"] = (
        final_df["last_gender"]
        .fillna(gender_mode)
        .apply(lambda x: 0 if x == "Ж" else 1)
    )

    final_df["main_part_of_day"] = final_df["main_part_of_day"].apply(part_of_day_e)

    final_df = final_df.drop(columns=["first_dt", "last_dt"])

    # 4) merge churn
    final_df = final_df.merge(
        last_activity[["Идентификатор устройства", "churn"]],
        how="left",
        on="Идентификатор устройства",
    )

    # 5) нормализация производителей
    vc = final_df["last_manufacturer"].value_counts()
    final_df["last_manufacturer"] = final_df["last_manufacturer"].where(
        final_df["last_manufacturer"].isin(vc[vc >= 5000].index),
        "Другие",
    )

    return final_df


# ==============================
# Обучение моделей и метрики
# ==============================

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    is_catboost: bool = False,
    cat_features: List[str] | None = None,
) -> Dict[str, Any]:
    start_time = time.time()

    if is_catboost:
        model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

    train_time = time.time() - start_time
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Train_time_sec": train_time,
    }


def train_and_evaluate_models(final_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    TARGET = "churn"
    ID_COL = "Идентификатор устройства"

    if TARGET not in final_df.columns:
        raise ValueError("В final_df нет колонки 'churn'")

    X = final_df.drop(columns=[TARGET, ID_COL])
    y = final_df[TARGET]

    cat_features = ["last_manufacturer"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # one-hot для LGBM / XGB / RF
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, axis=1, fill_value=0)

    metrics: Dict[str, Dict[str, Any]] = {}

    # CatBoost
    cat_model = CatBoostClassifier(
        learning_rate=0.03,
        l2_leaf_reg=3,
        iterations=800,
        depth=4,
        bagging_temperature=2,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
    )

    metrics["CatBoost"] = evaluate_model(
        cat_model,
        X_train,
        X_test,
        y_train,
        y_test,
        is_catboost=True,
        cat_features=cat_features,
    )

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )

    metrics["LightGBM"] = evaluate_model(
        lgb_model,
        X_train_enc,
        X_test_enc,
        y_train,
        y_test,
    )

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    metrics["XGBoost"] = evaluate_model(
        xgb_model,
        X_train_enc,
        X_test_enc,
        y_train,
        y_test,
    )

    # RandomForest
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.6,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )

    metrics["RandomForest"] = evaluate_model(
        rf_model,
        X_train_enc,
        X_test_enc,
        y_train,
        y_test,
    )

    n_train = int(len(y_train))
    n_test = int(len(y_test))
    churn_share = float(y.mean())

    for m in metrics.values():
        m["n_train"] = n_train
        m["n_test"] = n_test
        m["churn_share"] = churn_share

    return metrics


# ==============================
# Графики (Plotly) — БЕЗ data_frame для bar
# ==============================

def generate_plots(
    data: pd.DataFrame,
    socdem: pd.DataFrame,
    final_df: pd.DataFrame,
) -> List[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_urls: List[str] = []
    template = "plotly_white"

    # Рабочая копия событий
    df = data.copy()

    # 0. При необходимости — восстановить номера сессий
    if "Дата и время события" in df.columns and "Идентификатор устройства" in df.columns:
        if "Номер сессии в рамках устройства" not in df.columns:
            df = df.sort_values(["Идентификатор устройства", "Дата и время события"])
            df["prev_time"] = df.groupby("Идентификатор устройства")["Дата и время события"].shift(1)
            df["gap_sec"] = (df["Дата и время события"] - df["prev_time"]).dt.total_seconds()
            df["Номер сессии в рамках устройства"] = (df["gap_sec"] > 1800).cumsum()
            df.drop(columns=["prev_time", "gap_sec"], inplace=True)

    # 1. Распределение churn (0/1)
    if "churn" in final_df.columns:
        vc = final_df["churn"].value_counts().sort_index()  # 0,1
        churn_values = vc.index.astype(str)
        churn_counts = vc.values

        fig = px.bar(
            x=churn_values,
            y=churn_counts,
            title="Распределение оттока (0 и 1)",
            template=template,
            labels={"x": "churn", "y": "count"},
        )
        fig.update_xaxes(type="category")

        fname = PLOTS_DIR / "churn_blocks.html"
        pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
        plot_urls.append(f"/static/plots/{fname.name}")

    # 2. Распределение возраста по группам (из age_back)
    if "age_back" in df.columns:
        bins = [0, 18, 25, 35, 45, 60, 120]
        labels = ["<18", "18–24", "25–34", "35–44", "45–59", "60+"]

        df_age = df.dropna(subset=["age_back"]).copy()
        if not df_age.empty:
            df_age["age_group"] = pd.cut(df_age["age_back"], bins=bins, labels=labels, right=False)
            age_counts = (
                df_age["age_group"]
                .value_counts()
                .reindex(labels)
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            age_counts.columns = ["age_group", "count"]

            fig = px.bar(
                x=age_counts["age_group"],
                y=age_counts["count"],
                title="Распределение пользователей по возрастным группам",
                template=template,
                labels={"x": "Возрастная группа", "y": "Количество событий"},
            )

            fname = PLOTS_DIR / "age_groups_overall.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

    # 3. Активность по часам суток
    if "Дата и время события" in df.columns:
        df_hour = df.copy()
        df_hour["hour"] = df_hour["Дата и время события"].dt.hour

        hour_counts = (
            df_hour.groupby("hour")
            .size()
            .reset_index(name="count")
            .sort_values("hour")
        )

        fig = px.bar(
            x=hour_counts["hour"],
            y=hour_counts["count"],
            title="Распределение активности по часам суток",
            template=template,
            labels={"x": "Час", "y": "Количество событий"},
        )
        fname = PLOTS_DIR / "activity_by_hour.html"
        pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
        plot_urls.append(f"/static/plots/{fname.name}")

        # 4. Активность по часам + пол
        if "gender" in df_hour.columns:
            hour_gender = (
                df_hour.groupby(["hour", "gender"])
                .size()
                .reset_index(name="count")
            )

            fig = px.line(
                hour_gender,
                x="hour",
                y="count",
                color="gender",
                markers=True,
                title="Активность по часам суток по полу",
                template=template,
            )
            fname = PLOTS_DIR / "activity_by_hour_gender.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

        # 5. Активность по часам + ОС
        if "ОС" in df_hour.columns:
            hour_os = (
                df_hour.groupby(["hour", "ОС"])
                .size()
                .reset_index(name="count")
            )
            fig = px.line(
                hour_os,
                x="hour",
                y="count",
                color="ОС",
                markers=True,
                title="Активность по часам суток по операционным системам",
                template=template,
            )
            fname = PLOTS_DIR / "activity_by_hour_os.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

    # 6. Длина сессий по возрастным группам
    if {"age_back", "Идентификатор устройства", "Номер сессии в рамках устройства"}.issubset(df.columns):
        session_sizes = (
            df.groupby(["Идентификатор устройства", "Номер сессии в рамках устройства", "age_back"])
            .size()
            .reset_index(name="session_len")
        )

        bins = [0, 18, 25, 35, 45, 60, 120]
        labels = ["<18", "18–24", "25–34", "35–44", "45–59", "60+"]

        session_sizes["age_group"] = pd.cut(
            session_sizes["age_back"], bins=bins, labels=labels, right=False
        )

        fig = px.box(
            session_sizes,
            x="age_group",
            y="session_len",
            title="Длина сессий по возрастным группам (лог масштаб)",
            template=template,
            color="age_group",
        )
        fig.update_yaxes(type="log")

        fname = PLOTS_DIR / "session_length_by_age.html"
        pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
        plot_urls.append(f"/static/plots/{fname.name}")

        # 7. Длина сессий по ОС
        if "ОС" in df.columns:
            session_os = (
                df.groupby(["Идентификатор устройства", "Номер сессии в рамках устройства", "ОС"])
                .size()
                .reset_index(name="session_len")
            )
            fig = px.box(
                session_os,
                x="ОС",
                y="session_len",
                title="Длина сессий по операционным системам (лог масштаб)",
                template=template,
                color="ОС",
            )
            fig.update_yaxes(type="log")

            fname = PLOTS_DIR / "session_length_by_os.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

    # 8. Активность по производителям устройств
    if "Производитель устройства" in df.columns:
        vc = (
            df["Производитель устройства"]
            .value_counts()
            .head(20)
            .reset_index()
        )
        vc.columns = ["Производитель устройства", "count"]

        fig = px.bar(
            x=vc["Производитель устройства"],
            y=vc["count"],
            title="Активность по топ-20 производителям устройств",
            template=template,
            labels={"x": "Производитель", "y": "Количество событий"},
        )
        fig.update_xaxes(tickangle=-45)

        fname = PLOTS_DIR / "top_vendors.html"
        pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
        plot_urls.append(f"/static/plots/{fname.name}")

        # 9. Возрастные группы по производителям
        if "age_back" in df.columns:
            bins = [0, 18, 25, 35, 45, 60, 120]
            labels = ["<18", "18–24", "25–34", "35–44", "45–59", "60+"]

            df_v = df[df["Производитель устройства"].isin(vc["Производитель устройства"])].copy()
            df_v["age_group"] = pd.cut(df_v["age_back"], bins=bins, labels=labels, right=False)

            age_vendor = (
                df_v.groupby(["Производитель устройства", "age_group"])
                .size()
                .reset_index(name="count")
            )

            fig = px.bar(
                age_vendor,
                x="Производитель устройства",
                y="count",
                color="age_group",
                barmode="stack",
                title="Распределение возрастных групп по производителям устройств",
                template=template,
            )
            fig.update_xaxes(tickangle=-45)

            fname = PLOTS_DIR / "age_group_by_vendor.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

    # 10. Количество сессий по моделям устройств
    if {"Модель устройства", "Номер сессии в рамках устройства"}.issubset(df.columns):
        sessions_per_model = (
            df.groupby("Модель устройства")["Номер сессии в рамках устройства"]
            .nunique()
            .reset_index(name="n_sessions")
        )
        sessions_per_model = sessions_per_model.sort_values("n_sessions", ascending=False).head(20)

        fig = px.bar(
            sessions_per_model,
            x="n_sessions",
            y="Модель устройства",
            orientation="h",
            title="Количество сессий по моделям устройств (топ-20)",
            template=template,
        )

        fname = PLOTS_DIR / "sessions_per_model.html"
        pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
        plot_urls.append(f"/static/plots/{fname.name}")

    # 11. Функционал × возрастные группы (heatmap)
    if {"Функционал", "age_back"}.issubset(df.columns):
        df_fun = df.copy()
        df_fun = df_fun[df_fun["Функционал"] != "Открытие экрана"]
        df_fun = df_fun.dropna(subset=["age_back"])

        bins = [0, 18, 25, 35, 45, 60, 120]
        labels = ["<18", "18–24", "25–34", "35–44", "45–59", "60+"]

        if not df_fun.empty:
            df_fun["age_group"] = pd.cut(df_fun["age_back"], bins=bins, labels=labels, right=False)

            pivot = (
                df_fun.pivot_table(
                    index="Функционал",
                    columns="age_group",
                    values="Идентификатор устройства",
                    aggfunc="nunique",
                    fill_value=0,
                )
            )

            # Ограничим количеством функционалов (топ-20 по сумме, чтобы картинка была читабельной)
            top_funcs = pivot.sum(axis=1).sort_values(ascending=False).head(20).index
            pivot = pivot.loc[top_funcs]

            fig = px.imshow(
                pivot,
                labels=dict(x="Возрастная группа", y="Функционал", color="Кол-во уникальных устройств"),
                title="Активность функционала по возрастным группам (топ-20 функционалов)",
                template=template,
                aspect="auto",
            )

            fname = PLOTS_DIR / "functional_by_age_heatmap.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

    # 12. Распределение возрастов по полу (по словарю соцдема)
    if socdem is not None and {"age_back", "gender"}.issubset(socdem.columns):
        fig = px.histogram(
            socdem,
            x="age_back",
            color="gender",
            barmode="stack",
            nbins=20,
            title="Распределение возрастов по полу (по словарю соцдема)",
            template=template,
        )
        fname = PLOTS_DIR / "age_by_gender.html"
        pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
        plot_urls.append(f"/static/plots/{fname.name}")

    # 13. Корреляционная матрица числовых признаков final_df
    if not final_df.empty:
        numeric_cols = final_df.select_dtypes(include=[np.number])
        if numeric_cols.shape[1] >= 2:
            corr_matrix = numeric_cols.corr()

            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Корреляционная матрица числовых признаков",
                template=template,
            )

            fname = PLOTS_DIR / "corr_heatmap.html"
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs="cdn")
            plot_urls.append(f"/static/plots/{fname.name}")

    return plot_urls


# ==============================
# Точка входа для FastAPI
# ==============================

def process_uploaded_files(
    data_file: UploadFile,
    dct_file: UploadFile,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    events = _read_uploaded_table(data_file)
    socdem = _read_uploaded_table(dct_file)

    data = preprocess_data(events, socdem)
    final_df = build_user_features(data)

    metrics = train_and_evaluate_models(final_df)
    # ВАЖНО: для графиков используем уже очищенные data, а не сырые events
    plots = generate_plots(data, socdem, final_df)

    return metrics, plots

