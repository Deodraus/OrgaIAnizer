"""Configuración centralizada del proyecto de clasificación y organización local.

Optimizada para equipos de recursos limitados (CPU con oneDNN, desde 4 GB de RAM):
streaming desde disco, lotes pequeños y modo bajo consumo (`--on-demand`).
Puedes personalizar rutas y palabras clave copiando `user_rules.example.json`
a `user_rules.json` (ignorado por Git).
"""

import json
from pathlib import Path

# --- Rutas principales ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
INBOX_DIR: Path = PROJECT_ROOT / "inbox"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODELS_STAGING_DIR: Path = MODELS_DIR / "staging"
MODELS_PROD_DIR: Path = MODELS_DIR / "production"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EVALUATIONS_DIR: Path = REPORTS_DIR / "evaluations"

LOGS_DIR: Path = PROJECT_ROOT / "logs"
TENSORBOARD_DIR: Path = LOGS_DIR / "tensorboard"

DATASET_SUMMARY_JSON: Path = REPORTS_DIR / "dataset_summary.json"
DATASET_DISTRIBUTION_PNG: Path = REPORTS_DIR / "dataset_distribution.png"

# --- Ramas del dataset ---
# Nivel 1: separación determinista por tipo de medio (imágenes vs videos)
MULTIMEDIA_BRANCHES: tuple[str, ...] = ("imagenes", "videos")
# Ramas reservadas para el plan integral futuro (solo reporte informativo hoy)
INFORMATIVE_BRANCHES: tuple[str, ...] = ("documentos", "descargas")

# --- Extensiones soportadas (en minúsculas) ---
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4"})
SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Extensiones no multimedia (Documentos, Programas, Comprimidos, Cursores)
EXECUTABLE_EXTENSIONS: frozenset[str] = frozenset({".exe", ".msi", ".msix", ".appx"})
ARCHIVE_EXTENSIONS: frozenset[str] = frozenset(
    {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
)
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
WORD_EXTENSIONS: frozenset[str] = frozenset({".docx", ".doc", ".odt", ".rtf"})
CURSOR_EXTENSIONS: frozenset[str] = frozenset({".ani", ".cur"})
NON_MEDIA_EXTENSIONS: frozenset[str] = (
    EXECUTABLE_EXTENSIONS
    | ARCHIVE_EXTENSIONS
    | PDF_EXTENSIONS
    | WORD_EXTENSIONS
    | CURSOR_EXTENSIONS
)

# Rutas destino relativas (respetando exactamente tu estructura en dataset/)
DEST_PROGRAMAS_REL: str = "descargas/programas"
DEST_COMPRIMIDOS_REL: str = "descargas/comprimidos(zip,rar,etc)"
DEST_DESCOMPRIMIDOS_REL: str = "descargas/descomprimidos"
DEST_PDFS_REL: str = "documentos/PDFs"
DEST_WORDS_REL: str = "documentos/WordS"
DEST_UNIVERSIDAD_REL: str = "documentos/Universidad"
DEST_MALLAS_REL: str = "documentos/Universidad/Mallas Curriculares"
DEST_CURSORES_REL: str = "documentos/Mi_Cursor"

# Descompresión automática de archivos comprimidos
AUTO_EXTRACT_ARCHIVES: bool = True

# Archivos de sistema de Windows/macOS a ignorar en el conteo de clases
IGNORED_SYSTEM_FILES: frozenset[str] = frozenset(
    {"desktop.ini", "thumbs.db", ".ds_store", "ehthumbs.db"}
)

# --- Parámetros del modelo y hardware (3.44 GB RAM usable) ---
IMAGE_SIZE: tuple[int, int] = (224, 224)
RESIZE_MODE: str = "letterbox"  # Preserva relación de aspecto (clave para 16:9 vs 1:1)
LETTERBOX_PAD_COLOR: tuple[int, int, int] = (128, 128, 128)  # Gris neutro (~0.0 en [-1, 1])
BACKBONE_NAME: str = "MobileNetV2"
MOBILENET_ALPHA: float = 1.0  # Puede bajarse a 0.5 si se requiere aún menor consumo de RAM

# Extracción de fotogramas en videos (.mp4) y animaciones (.gif/.webp)
VIDEO_FRAMES_PER_FILE: int = 3  # Rango especificado: 3 a 5 fotogramas
VIDEO_FRAME_START_RATIO: float = 0.10
VIDEO_FRAME_END_RATIO: float = 0.85

# Entrenamiento supervisado (ajustado tras inspección del Paso 2)
BATCH_SIZE: int = 8
MAX_SAMPLES_PER_CLASS: int = 250  # Reduce desbalanceo de imagenes/ (851 vs 41) y acelera CPU
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
RANDOM_SEED: int = 42

# Hiperparámetros de entrenamiento en 2 fases (CPU-friendly)
EPOCHS_PHASE1: int = 15
LEARNING_RATE_PHASE1: float = 1e-3
EPOCHS_PHASE2_FINETUNE: int = 8
LEARNING_RATE_PHASE2: float = 1e-5
FINETUNE_UNFREEZE_LAYERS: int = 25
EARLY_STOPPING_PATIENCE: int = 4
DROPOUT_RATE: float = 0.30

# Rutas deterministas opcionales (nodos con 1 sola subcarpeta poblada).
# Personalizable en `user_rules.json` (ver `user_rules.example.json`).
DETERMINISTIC_ROUTES: dict[str, str] = {}

# Palabras clave opcionales de instituciones universitarias para clasificar documentos.
# Personalizable en `user_rules.json`.
UNI_INSTITUTION_KEYWORDS: tuple[str, ...] = ()

# Manifiesto de modelos promovidos a producción y rutas jerárquicas
HIERARCHY_MANIFEST_PATH: Path = MODELS_PROD_DIR / "hierarchy_manifest.json"

# Límite seguro para decodificar fotos de alta resolución (ej. cámara 108 MP) sin saturar RAM
MAX_IMAGE_PIXELS_ALLOWED: int = 250_000_000

# --- Carga opcional de reglas personalizadas locales (`user_rules.json`, fuera de Git) ---
USER_RULES_PATH: Path = PROJECT_ROOT / "user_rules.json"
if USER_RULES_PATH.exists():
    try:
        with USER_RULES_PATH.open("r", encoding="utf-8") as _fh:
            _user_rules = json.load(_fh)
        if isinstance(_user_rules.get("deterministic_routes"), dict):
            DETERMINISTIC_ROUTES.update(
                {str(k): str(v) for k, v in _user_rules["deterministic_routes"].items()}
            )
        if isinstance(_user_rules.get("university_institution_keywords"), list):
            UNI_INSTITUTION_KEYWORDS = tuple(
                str(k).strip().lower()
                for k in _user_rules["university_institution_keywords"]
                if str(k).strip()
            )
        _dests = _user_rules.get("destinations")
        if isinstance(_dests, dict):
            DEST_PROGRAMAS_REL = str(_dests.get("programas", DEST_PROGRAMAS_REL))
            DEST_COMPRIMIDOS_REL = str(_dests.get("comprimidos", DEST_COMPRIMIDOS_REL))
            DEST_DESCOMPRIMIDOS_REL = str(_dests.get("descomprimidos", DEST_DESCOMPRIMIDOS_REL))
            DEST_PDFS_REL = str(_dests.get("pdfs", DEST_PDFS_REL))
            DEST_WORDS_REL = str(_dests.get("words", DEST_WORDS_REL))
            DEST_UNIVERSIDAD_REL = str(_dests.get("universidad", DEST_UNIVERSIDAD_REL))
            DEST_MALLAS_REL = str(_dests.get("mallas", DEST_MALLAS_REL))
            DEST_CURSORES_REL = str(_dests.get("cursores", DEST_CURSORES_REL))
    except Exception:
        pass

# Motor de vigilancia (Paso 4 + soporte multi-carpeta)
CONFIDENCE_THRESHOLD: float = 0.80
DRY_RUN_DEFAULT: bool = True
ON_DEMAND_DEBOUNCE_SECONDS: float = 2.0  # Agrupa capturas/descargas seguidas antes de cargar TF
WATCHER_PID_FILE: Path = LOGS_DIR / "watcher.pid"

# Carpetas reales de Windows vigiladas en el perfil `--profile pc`
DEFAULT_PC_WATCH_DIRS: tuple[Path, ...] = (
    Path.home() / "Pictures" / "Screenshots",
    Path.home() / "Downloads",
)

# Umbrales de diagnóstico de balanceo (Paso 2)
MIN_SAMPLES_FOR_STRATIFIED_SPLIT: int = 7   # Mínimo teórico para 70/15/15 (>=1 val, >=1 test)
MIN_RECOMMENDED_SAMPLES: int = 20           # Mínimo sano para generalización y evaluación estable
MODERATE_IMBALANCE_RATIO: float = 2.5
HIGH_IMBALANCE_RATIO: float = 5.0
