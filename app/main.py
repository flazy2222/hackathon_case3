from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse

from .nlp_pipeline import process_nlp_file

from .churn_pipeline import process_uploaded_files
from .config import STATIC_DIR

app = FastAPI(
    title="Churn Analytics Service",
    description="Service for churn & NLP analytics in ЖКХ app.",
)

# Подключаем статику (графики и т.п.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Подключаем шаблоны
templates = Jinja2Templates(directory="templates")
from fastapi.responses import HTMLResponse
import os

@app.get("/graphs", response_class=HTMLResponse)
def graphs_page():
    files = os.listdir("static/plots")
    links = "".join([f'<li><a href="/static/plots/{f}" target="_blank">{f}</a></li>' for f in files])
    return f"""
    <h1>Графики анализа данных</h1>
    <ul>
        {links}
    </ul>
    """

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    data_file: UploadFile = File(..., description="Основной лог событий CSV"),
    dct_file: UploadFile = File(..., description="Словарь соцдема CSV"),
):
    # Основной пайплайн анализа
    metrics, plots = process_uploaded_files(data_file, dct_file)

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "metrics": metrics,
            "plots": plots,
        },
    )

@app.get("/nlp", response_class=HTMLResponse)
async def nlp_page():
    """
    Простая страница для загрузки файла с обращениями
    и просмотра NLP-аналитики.
    """
    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <title>NLP-аналитика обращений</title>
        <link rel="stylesheet" href="/static/style.css" />
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background-color: #f5f7fb;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 1100px;
                margin: 40px auto;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(15,23,42,0.08);
                padding: 24px 32px 32px;
            }
            h1 {
                margin-top: 0;
                font-size: 26px;
                color: #111827;
            }
            p.subtitle {
                margin-top: 4px;
                color: #6b7280;
            }
            .upload-block {
                margin: 24px 0;
                padding: 16px 20px;
                border-radius: 12px;
                border: 1px dashed #9ca3af;
                background: #f9fafb;
            }
            .upload-block label {
                font-weight: 600;
                color: #374151;
            }
            .upload-block input[type="file"] {
                margin-top: 8px;
            }
            .btn-primary {
                margin-top: 16px;
                display: inline-block;
                padding: 8px 18px;
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, #2563eb, #10b981);
                color: #ffffff;
                font-weight: 600;
                cursor: pointer;
            }
            .btn-primary:hover {
                opacity: 0.92;
            }
            .nav-back {
                margin-top: 24px;
            }
            .nav-back a {
                color: #2563eb;
                text-decoration: none;
                font-size: 14px;
            }
            .nav-back a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NLP-аналитика обращений</h1>
            <p class="subtitle">
                Загрузите файл с обращениями (CSV или Excel), по которым уже посчитаны предсказания моделей.
                Сервис автоматически построит распределения по уровням негатива и типам обращений.
            </p>

            <div class="upload-block">
                <form action="/upload_nlp" method="post" enctype="multipart/form-data">
                    <label for="nlp_file">Файл с обращениями:</label><br/>
                    <input type="file" id="nlp_file" name="nlp_file" accept=".csv,.xlsx,.xls" required />
                    <br/>
                    <button type="submit" class="btn-primary">Построить NLP-отчёт</button>
                </form>
            </div>

            <div class="nav-back">
                <a href="/">< Вернуться к анализу поведения пользователей</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/upload_nlp", response_class=HTMLResponse)
async def upload_nlp(nlp_file: UploadFile = File(...)):
    """
    Принимаем файл с обращениями, запускаем NLP-пайплайн и
    возвращаем HTML-страницу с метриками и интерактивными графиками.
    """
    metrics, plots = process_nlp_file(nlp_file)

    # Раскладываем метрики для удобного отображения
    ds = metrics.get("dataset_info", {})
    sm = metrics.get("sentiment_model", {})
    tm = metrics.get("topic_model", {})

    # Формируем HTML
    rows_html = ""
    for name, val in [
        ("Строк в датасете", ds.get("n_rows")),
        ("Столбцов в датасете", ds.get("n_columns")),
        ("Колонка уровня негатива", ds.get("sentiment_column")),
        ("Колонка типа обращения", ds.get("topic_column")),
    ]:
        rows_html += f"<tr><td>{name}</td><td>{val}</td></tr>"

    plots_html = ""
    for url in plots:
        plots_html += f"""
        <div class="plot-card">
            <iframe src="{url}" loading="lazy"></iframe>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <title>NLP-аналитика обращений</title>
        <style>
            body {{
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background-color: #f5f7fb;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 1200px;
                margin: 40px auto;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(15,23,42,0.08);
                padding: 24px 32px 32px;
            }}
            h1 {{
                margin-top: 0;
                font-size: 26px;
                color: #111827;
            }}
            p.subtitle {{
                margin-top: 4px;
                color: #6b7280;
            }}
            table.info-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 16px;
                margin-bottom: 24px;
            }}
            table.info-table th,
            table.info-table td {{
                border: 1px solid #e5e7eb;
                padding: 8px 10px;
                font-size: 14px;
            }}
            table.info-table th {{
                background: #f9fafb;
                text-align: left;
            }}
            .model-block {{
                margin-bottom: 24px;
                padding: 16px 20px;
                border-radius: 12px;
                background: #f9fafb;
            }}
            .model-block h2 {{
                margin: 0 0 8px 0;
                font-size: 18px;
                color: #111827;
            }}
            .model-block p {{
                margin: 4px 0;
                font-size: 14px;
                color: #4b5563;
            }}
            .plots-grid {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
                gap: 32px;
                margin-top: 16px;
            }}
            .plot-card {{
                background: #f9fafb;
                border-radius: 12px;
                padding: 8px;
                box-shadow: 0 4px 12px rgba(15,23,42,0.06);
            }}
            .plot-card iframe {{
                width: 100%;
                height: 560px;
                border: none;
                border-radius: 8px;
                background: #ffffff;
            }}
            .nav-back {{
                margin-top: 24px;
            }}
            .nav-back a {{
                color: #2563eb;
                text-decoration: none;
                font-size: 14px;
            }}
            .nav-back a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NLP-аналитика обращений</h1>
            <p class="subtitle">
                Итоги обработки загруженного датасета с обращениями и предсказаниями моделей.
            </p>

            <h2>Информация о датасете</h2>
            <table class="info-table">
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div class="model-block">
                <h2>Модель оценки уровня негатива</h2>
                <p>{sm.get("description", "")}</p>
                <p><b>Классы:</b> {sm.get("classes")}</p>
                <p><b>F1 macro (валидация):</b> {sm.get("f1_macro_val")}</p>
                <p><b>F1 macro (тест):</b> {sm.get("f1_macro_test")}</p>
                <p><b>Комментарий:</b> {sm.get("notes")}</p>
            </div>

            <div class="model-block">
                <h2>Модель классификации типов обращений</h2>
                <p>{tm.get("description", "")}</p>
                <p><b>Классы:</b> {tm.get("classes")}</p>
                <p><b>F1 macro (валидация):</b> {tm.get("f1_macro_val")}</p>
                <p><b>F1 macro (тест):</b> {tm.get("f1_macro_test")}</p>
                <p><b>Комментарий:</b> {tm.get("notes")}</p>
            </div>

            <h2>Графики по обращениям</h2>
            <div class="plots-grid">
                {plots_html}
            </div>

            <div class="nav-back">
                <a href="/nlp">< Загрузить другой файл</a><br/>
                <a href="/">< Вернуться к анализу поведения пользователей</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/nlp_dashboard", response_class=HTMLResponse)
async def nlp_dashboard_page():
    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <title>NLP-дашборд обращений</title>
        <link rel="stylesheet" href="/static/style.css" />
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background-color: #f5f7fb;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 1100px;
                margin: 40px auto;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(15,23,42,0.08);
                padding: 24px 32px 32px;
            }
            h1 {
                margin-top: 0;
                font-size: 26px;
                color: #111827;
            }
            p.subtitle {
                margin-top: 4px;
                color: #6b7280;
            }
            .upload-block {
                margin: 24px 0;
                padding: 16px 20px;
                border-radius: 12px;
                border: 1px dashed #9ca3af;
                background: #f9fafb;
            }
            .upload-block label {
                font-weight: 600;
                color: #374151;
            }
            .upload-block input[type="file"] {
                margin-top: 8px;
            }
            .btn-primary {
                margin-top: 16px;
                display: inline-block;
                padding: 8px 18px;
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, #2563eb, #10b981);
                color: #ffffff;
                font-weight: 600;
                cursor: pointer;
            }
            .btn-primary:hover {
                opacity: 0.92;
            }
            .nav-back {
                margin-top: 24px;
            }
            .nav-back a {
                color: #2563eb;
                text-decoration: none;
                font-size: 14px;
            }
            .nav-back a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NLP-дашборд обращений</h1>
            <p class="subtitle">
                Загрузите файл с обращениями (CSV или Excel) с предсказаниями моделей.
                Сервис посчитает KPI и построит графики.
            </p>

            <div class="upload-block">
                <form action="/upload_nlp_dashboard" method="post" enctype="multipart/form-data">
                    <label for="nlp_file">Файл с обращениями:</label><br/>
                    <input type="file" id="nlp_file" name="nlp_file" accept=".csv,.xlsx,.xls" required />
                    <br/>
                    <button type="submit" class="btn-primary">Построить отчёт</button>
                </form>
            </div>

            <div class="nav-back">
                <a href="/">< Вернуться к анализу поведения пользователей</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/upload_nlp_dashboard", response_class=HTMLResponse)
async def upload_nlp_dashboard(nlp_file: UploadFile = File(...)):
    metrics, plots = process_nlp_file(nlp_file)

    ds = metrics.get("dataset_info", {})
    sm = metrics.get("sentiment_model", {})
    tm = metrics.get("topic_model", {})
    agg = metrics.get("aggregates", {})
    sentiment_dist = agg.get("sentiment_dist") or {}
    topic_dist = agg.get("topic_dist") or {}
    aggressive_share = agg.get("aggressive_share")

    def _fmt_pct(x):
        if x is None:
            return "—"
        return f"{x * 100:.1f}%"

    def _dist_to_html(dist, labels_map):
        if not dist:
            return "<i>нет данных</i>"
        items = []
        for k, info in dist.items():
            label = labels_map.get(k, str(k))
            pct = _fmt_pct(info.get("share"))
            cnt = info.get("count", 0)
            items.append(f"<li>{label}: {pct} ({cnt} шт.)</li>")
        return "<ul>" + "".join(items) + "</ul>"

    sentiment_labels = {
        0: "0 — нейтрально",
        1: "1 — эмоционально, без агрессии",
        2: "2 — агрессия",
    }
    topic_labels = {
        0: "0 — просьба / нейтральное действие",
        1: "1 — вопрос, не понял / не нашёл",
        2: "2 — жалоба / проблема",
        3: "3 — жалоба / проблема",
    }

    sentiment_dist_html = _dist_to_html(sentiment_dist, sentiment_labels)
    topic_dist_html = _dist_to_html(topic_dist, topic_labels)

    rows_html = ""
    for name, val in [
        ("Строк в датасете", ds.get("n_rows")),
        ("Столбцов в датасете", ds.get("n_columns")),
        ("Колонка уровня негатива", ds.get("sentiment_column")),
        ("Колонка типа обращения", ds.get("topic_column")),
    ]:
        rows_html += f"<tr><td>{name}</td><td>{val}</td></tr>"

    plots_html = ""
    for url in plots:
        plots_html += f"""
        <div class="plot-card">
            <iframe src="{url}" loading="lazy"></iframe>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8" />
        <title>NLP-дашборд обращений</title>
        <style>
            body {{
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background-color: #f5f7fb;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 1200px;
                margin: 40px auto;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(15,23,42,0.08);
                padding: 24px 32px 32px;
            }}
            h1 {{
                margin-top: 0;
                font-size: 26px;
                color: #111827;
            }}
            p.subtitle {{
                margin-top: 4px;
                color: #6b7280;
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 32px;
                margin: 20px 0 8px 0;
            }}
            .kpi-card {{
                background: #f9fafb;
                border-radius: 12px;
                padding: 12px 14px;
                box-shadow: 0 4px 10px rgba(15,23,42,0.04);
            }}
            .kpi-title {{
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: .08em;
                color: #6b7280;
                margin-bottom: 6px;
            }}
            .kpi-value {{
                font-size: 22px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 4px;
            }}
            .kpi-subtext {{
                font-size: 12px;
                color: #6b7280;
            }}
            table.info-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 16px;
                margin-bottom: 24px;
            }}
            table.info-table th,
            table.info-table td {{
                border: 1px solid #e5e7eb;
                padding: 8px 10px;
                font-size: 14px;
            }}
            table.info-table th {{
                background: #f9fafb;
                text-align: left;
            }}
            .model-block {{
                margin-bottom: 24px;
                padding: 16px 20px;
                border-radius: 12px;
                background: #f9fafb;
            }}
            .model-block h2 {{
                margin: 0 0 8px 0;
                font-size: 18px;
                color: #111827;
            }}
            .model-block p {{
                margin: 4px 0;
                font-size: 14px;
                color: #4b5563;
            }}
            .plots-grid {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
                gap: 32px;
                margin-top: 16px;
            }}
            .plot-card {{
                background: #f9fafb;
                border-radius: 12px;
                padding: 8px;
                box-shadow: 0 4px 12px rgba(15,23,42,0.06);
            }}
            .plot-card iframe {{
                width: 100%;
                height: 560px;
                border: none;
                border-radius: 8px;
                background: #ffffff;
            }}
            .nav-back {{
                margin-top: 24px;
            }}
            .nav-back a {{
                color: #2563eb;
                text-decoration: none;
                font-size: 14px;
            }}
            .nav-back a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NLP-дашборд обращений</h1>
            <p class="subtitle">
                Итоги обработки загруженного датасета с обращениями и предсказаниями моделей.
            </p>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-title">Всего обращений</div>
                    <div class="kpi-value">{ds.get("n_rows", "—")}</div>
                    <div class="kpi-subtext">Строк в датасете</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Проблемные / агрессивные</div>
                    <div class="kpi-value">{_fmt_pct(aggressive_share)}</div>
                    <div class="kpi-subtext">Доля обращений с агрессией или явной проблемой</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Колонки с метками</div>
                    <div class="kpi-value">sent: {ds.get("sentiment_column")}, type: {ds.get("topic_column")}</div>
                    <div class="kpi-subtext">Используются для анализа</div>
                </div>
            </div>

            <h2>Информация о датасете</h2>
            <table class="info-table">
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div class="model-block">
                <h2>Модель оценки уровня негатива</h2>
                <p>{sm.get("description", "")}</p>
                <p><b>Классы:</b> {sm.get("classes")}</p>
                <p><b>F1 macro (валидация):</b> {sm.get("f1_macro_val")}</p>
                <p><b>F1 macro (тест):</b> {sm.get("f1_macro_test")}</p>
                <p><b>Комментарий:</b> {sm.get("notes")}</p>
                <p><b>Распределение классов:</b></p>
                {sentiment_dist_html}
            </div>

            <div class="model-block">
                <h2>Модель классификации типов обращений</h2>
                <p>{tm.get("description", "")}</p>
                <p><b>Классы:</b> {tm.get("classes")}</p>
                <p><b>F1 macro (валидация):</b> {tm.get("f1_macro_val")}</p>
                <p><b>F1 macro (тест):</b> {tm.get("f1_macro_test")}</p>
                <p><b>Комментарий:</b> {tm.get("notes")}</p>
                <p><b>Распределение типов:</b></p>
                {topic_dist_html}
            </div>

            <h2>Графики по обращениям</h2>
            <div class="plots-grid">
                {plots_html}
            </div>

            <div class="nav-back">
                <a href="/nlp_dashboard">< Загрузить другой файл</a><br/>
                <a href="/">< Вернуться к анализу поведения пользователей</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
