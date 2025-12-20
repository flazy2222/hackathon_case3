from pathlib import Path

# Корневая директория проекта (churn_service/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Папка со статикой
STATIC_DIR = BASE_DIR / "static"

# Папка, куда будем сохранять графики
PLOTS_DIR = STATIC_DIR / "plots"

# На всякий случай — создаём папку, если её нет
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
