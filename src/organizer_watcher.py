r"""Paso 4+: Motor de Clasificación Jerárquica y Vigilancia Multi-Carpeta (`watchdog`).

Características principales:
- Vigila UNA o MÚLTIPLES carpetas simultáneamente (ej. `inbox/`, o `--profile pc` para
  `Pictures/Screenshots` + `Downloads`), manejando `on_created` y `on_moved`.
- Clasifica de forma jerárquica (Nivel 1 determinista por tipo de medio + modelos `.keras`
  promovidos en `models/production/` + rutas deterministas configuradas).
- Votación promediando probabilidades entre fotogramas extraídos (con soporte `letterbox`).
- Modo Bajo Consumo de RAM (`--on-demand`): el vigilante residente usa solo ~25 MB de RAM
  sin importar TensorFlow y lanza un subproceso efímero (sin ventana en Windows) únicamente
  cuando detecta archivos nuevos, liberando el 100% de la RAM de TensorFlow al terminar.
- Compatible con `pythonw.exe` (sin ventana de consola) y registro en `logs/organizer.log`.
- Modo Seguro (`--dry-run`) ACTIVADO POR DEFECTO: solo mueve archivos reales cuando se
  desactiva explícitamente (`--no-dry-run` o `--execute`) y la confianza >= umbral (80%).

Uso:
    # Modo seguro sobre inbox/ (por defecto)
    python -m src.organizer_watcher --once

    # Vigilar múltiples carpetas reales del PC (Screenshots + Downloads) en Modo Seguro
    python -m src.organizer_watcher --profile pc --on-demand

    # Vigilar carpetas personalizadas
    python -m src.organizer_watcher --watch-dir "C:\Users\<TU_USUARIO>\Pictures\Screenshots" --watch-dir "C:\Users\<TU_USUARIO>\Downloads"

    # Activar movimientos reales (cuando estés listo)
    python -m src.organizer_watcher --profile pc --on-demand --no-dry-run --threshold 0.80
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src import config
from src.media_extractor import extract_frames, get_media_type
from src.non_media_organizer import (
    classify_non_media_file,
    extract_archive_smart,
    get_non_media_category,
    inspect_archive,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Extensiones temporales de navegadores / gestores de descarga a ignorar hasta que terminen
TEMP_DOWNLOAD_EXTENSIONS: frozenset[str] = frozenset(
    {".crdownload", ".part", ".tmp", ".partial", ".download", ".opdownload", ".!ut"}
)


@dataclass
class LevelDecision:
    """Resultado de clasificación en un nivel de la jerarquía."""

    node_path: str
    predicted_label: str
    confidence: float
    method: str  # "deterministic_type" | "deterministic_route" | "neural_voting"
    num_frames: int = 1
    alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Resultado completo de clasificación jerárquica para un archivo."""

    source_path: Path
    status: str  # "READY" | "NO_MODEL" | "UNSUPPORTED" | "ERROR"
    target_rel_dir: str | None
    final_confidence: float
    decisions: list[LevelDecision] = field(default_factory=list)
    message: str = ""


def make_safe_console() -> Console:
    """Crea una consola `rich` segura incluso bajo `pythonw.exe` (donde stdout es None)."""
    if sys.stdout is None:
        return Console(file=open(os.devnull, "w", encoding="utf-8"), quiet=True)
    return Console()


def setup_logger(log_path: Path) -> logging.Logger:
    """Configura el logger persistente en `logs/organizer.log`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("media_organizer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)
    return logger


def should_ignore_path(path: Path) -> bool:
    """Filtra archivos de sistema, descargas temporales y formatos no soportados."""
    name_lower = path.name.lower()
    if name_lower in config.IGNORED_SYSTEM_FILES or name_lower.startswith("."):
        return True
    ext = path.suffix.lower()
    if ext in TEMP_DOWNLOAD_EXTENSIONS:
        return True
    if get_media_type(path) == "unsupported" and get_non_media_category(path) == "unsupported":
        return True
    return False


class HierarchicalClassifier:
    """Motor de inferencia jerárquica con carga perezosa y bajo consumo de RAM."""

    def __init__(
        self,
        manifest_path: Path = config.HIERARCHY_MANIFEST_PATH,
        models_dir: Path = config.MODELS_PROD_DIR,
        num_frames: int = config.VIDEO_FRAMES_PER_FILE,
        max_loaded_models: int = 2,
    ) -> None:
        self.manifest_path = manifest_path
        self.models_dir = models_dir
        self.num_frames = num_frames
        self.max_loaded_models = max(1, max_loaded_models)
        self._manifest: dict[str, Any] = self._load_manifest()
        self._model_cache: OrderedDict[str, tuple[Any, list[str]]] = OrderedDict()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "level1_media_routing": {"image": "imagenes", "video": "videos"},
                "deterministic_routes": dict(config.DETERMINISTIC_ROUTES),
                "models": {},
            }
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def reload_manifest(self) -> None:
        """Recarga el manifiesto de producción (útil si se promovieron nuevos modelos)."""
        self._manifest = self._load_manifest()

    @property
    def promoted_models(self) -> dict[str, Any]:
        return self._manifest.get("models", {})

    @property
    def deterministic_routes(self) -> dict[str, str]:
        return self._manifest.get("deterministic_routes", dict(config.DETERMINISTIC_ROUTES))

    def _get_model_and_classes(self, node_path: str) -> tuple[Any, list[str]] | None:
        """Carga el modelo `.keras` de un nodo con caché LRU acotada para 3.44 GB de RAM."""
        entry = self.promoted_models.get(node_path)
        if not entry:
            return None

        if node_path in self._model_cache:
            self._model_cache.move_to_end(node_path)
            return self._model_cache[node_path]

        model_file = self.models_dir / entry["model_file"]
        class_map_file = self.models_dir / entry["class_map_file"]
        if not model_file.exists():
            return None

        if class_map_file.exists():
            with class_map_file.open("r", encoding="utf-8") as fh:
                class_names = json.load(fh)["class_names"]
        else:
            class_names = list(entry["class_names"])

        # Liberar el modelo menos usado si se supera el límite de RAM configurado
        while len(self._model_cache) >= self.max_loaded_models:
            _, (old_model, _) = self._model_cache.popitem(last=False)
            del old_model
            gc.collect()

        import keras

        # compile=False acelera la carga, ahorra RAM y evita advertencias del optimizador
        model = keras.models.load_model(model_file, compile=False)
        self._model_cache[node_path] = (model, class_names)
        return model, class_names

    def classify_file(self, file_path: Path) -> ClassificationResult:
        """Clasifica un archivo bajando por el árbol jerárquico con votación por fotogramas."""
        media_type = get_media_type(file_path)
        if media_type == "unsupported":
            return ClassificationResult(
                source_path=file_path,
                status="UNSUPPORTED",
                target_rel_dir=None,
                final_confidence=0.0,
                message=f"Extensión no soportada ({file_path.suffix or 'sin extensión'})",
            )

        level1_map = self._manifest.get("level1_media_routing", {"image": "imagenes", "video": "videos"})
        current_node = level1_map.get(media_type, "imagenes" if media_type == "image" else "videos")
        decisions: list[LevelDecision] = [
            LevelDecision(
                node_path="nivel_1",
                predicted_label=current_node,
                confidence=1.0,
                method="deterministic_type",
                num_frames=0,
            )
        ]

        # Extraer fotogramas una sola vez (reutilizados en todos los niveles del árbol)
        try:
            frames = extract_frames(file_path, num_frames=self.num_frames, apply_preprocess=True)
        except Exception as exc:
            return ClassificationResult(
                source_path=file_path,
                status="ERROR",
                target_rel_dir=None,
                final_confidence=0.0,
                decisions=decisions,
                message=f"Error al extraer fotogramas: {exc}",
            )

        neural_confidences: list[float] = []
        visited_nodes: set[str] = set()

        while current_node not in visited_nodes:
            visited_nodes.add(current_node)

            # 1. Ruta determinista (ej. imagenes/galeria -> imagenes/galeria/fotos)
            if current_node in self.deterministic_routes:
                next_node = self.deterministic_routes[current_node]
                label = Path(next_node).name
                decisions.append(
                    LevelDecision(
                        node_path=current_node,
                        predicted_label=label,
                        confidence=1.0,
                        method="deterministic_route",
                        num_frames=len(frames),
                    )
                )
                current_node = next_node
                continue

            # 2. Modelo neuronal promovido para este nivel
            loaded = self._get_model_and_classes(current_node)
            if loaded is None:
                break

            model, class_names = loaded
            probs_per_frame = model.predict(frames, batch_size=len(frames), verbose=0)
            # Votación promediando probabilidades de todos los fotogramas extraídos
            mean_probs = np.mean(probs_per_frame, axis=0)
            sorted_idx = np.argsort(mean_probs)[::-1]
            best_idx = int(sorted_idx[0])
            best_conf = float(mean_probs[best_idx])
            best_label = class_names[best_idx]

            alternatives = [
                (class_names[int(i)], float(mean_probs[int(i)])) for i in sorted_idx[:3]
            ]
            decisions.append(
                LevelDecision(
                    node_path=current_node,
                    predicted_label=best_label,
                    confidence=best_conf,
                    method="neural_voting",
                    num_frames=len(frames),
                    alternatives=alternatives,
                )
            )
            neural_confidences.append(best_conf)
            current_node = f"{current_node}/{best_label}"

        if not neural_confidences:
            return ClassificationResult(
                source_path=file_path,
                status="NO_MODEL",
                target_rel_dir=None,
                final_confidence=0.0,
                decisions=decisions,
                message=(
                    f"No hay modelos promovidos en producción para la rama '{current_node}/'. "
                    "Entrena y aprueba modelos primero con `python -m src.train_supervised`."
                ),
            )

        # Confianza final conservadora: el cuello de botella (mínimo) entre los niveles neuronales
        final_conf = min(neural_confidences)
        return ClassificationResult(
            source_path=file_path,
            status="READY",
            target_rel_dir=current_node,
            final_confidence=final_conf,
            decisions=decisions,
            message="Clasificación jerárquica completada.",
        )


def wait_for_file_stability(path: Path, timeout_sec: float = 15.0, poll_interval: float = 0.5) -> bool:
    """Espera a que un archivo recién creado/movido termine de escribirse en Windows."""
    deadline = time.monotonic() + timeout_sec
    last_size = -1

    while time.monotonic() < deadline:
        if not path.exists() or not path.is_file():
            return False
        try:
            cur_size = path.stat().st_size
            if cur_size > 0 and cur_size == last_size:
                # Verificar permiso de lectura real (evita archivos bloqueados por el navegador)
                with path.open("rb") as fh:
                    fh.read(16)
                return True
            last_size = cur_size
        except OSError:
            last_size = -1
        time.sleep(poll_interval)

    return path.exists() and path.is_file() and path.stat().st_size > 0


def resolve_non_colliding_path(dest_dir: Path, filename: str) -> Path:
    """Genera una ruta destino sin sobrescribir jamás archivos existentes."""
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alt = dest_dir / f"{stem}_{counter}{suffix}"
        if not alt.exists():
            return alt
        counter += 1


def format_decision_trace(decisions: list[LevelDecision]) -> str:
    """Formatea la ruta de decisiones por nivel para consola y logs."""
    parts: list[str] = []
    for d in decisions:
        if d.method == "deterministic_type":
            parts.append(f"{d.predicted_label} (tipo)")
        elif d.method == "deterministic_route":
            parts.append(f"{d.predicted_label} (regla)")
        else:
            parts.append(f"{d.predicted_label} ({d.confidence:.1%}, {d.num_frames}f)")
    return " -> ".join(parts)


class MediaOrganizerService:
    """Coordina clasificación (multimedia + documentos/programas/comprimidos), Dry-Run y logging."""

    def __init__(
        self,
        classifier: HierarchicalClassifier | None,
        dest_root: Path,
        threshold: float,
        dry_run: bool,
        console: Console,
        logger: logging.Logger,
    ) -> None:
        self.classifier = classifier
        self.dest_root = dest_root.resolve()
        self.threshold = threshold
        self.dry_run = dry_run
        self.console = console
        self.logger = logger
        self._recent_events: dict[str, tuple[float, int, float]] = {}

    def _process_non_media(self, path: Path) -> ClassificationResult | None:
        """Procesa Documentos, Programas, Cursores y Comprimidos sin cargar TensorFlow."""
        decision = classify_non_media_file(path)
        if decision is None:
            return None

        target_dir = self.dest_root / Path(decision.target_rel_dir)
        target_path = resolve_non_colliding_path(target_dir, path.name)
        descomprimidos_dir = self.dest_root / Path(config.DEST_DESCOMPRIMIDOS_REL)

        if self.dry_run:
            extra_info = ""
            if decision.is_archive and config.AUTO_EXTRACT_ARCHIVES:
                insp = inspect_archive(path)
                if not insp.ok:
                    extra_info = f" | Descompresión: OMITIRÍA ({insp.error_message})"
                elif insp.has_single_root_folder:
                    extra_info = (
                        f" | Descomprimiría en: {config.DEST_DESCOMPRIMIDOS_REL}/{insp.single_root_name}/ "
                        f"(carpeta interna existente, {insp.file_count} archivo(s))"
                    )
                else:
                    extra_info = (
                        f" | Crearía carpeta y descomprimiría en: "
                        f"{config.DEST_DESCOMPRIMIDOS_REL}/{insp.wrapper_folder_name}/ "
                        f"({insp.top_level_count} sueltos en raíz, {insp.file_count} archivo(s))"
                    )

            self.console.print(
                f"[bold cyan][DRY-RUN | MOVERÍA][/bold cyan] [white]{path.parent.name}/{path.name}[/white] "
                f"-> [green]{decision.target_rel_dir}/[/green] "
                f"([magenta]{decision.reason}[/magenta]{extra_info})"
            )
            self.logger.info(
                "DRY_RUN_NON_MEDIA | %s -> %s | reason=%s%s",
                path,
                target_path,
                decision.reason,
                extra_info,
            )
            return ClassificationResult(
                source_path=path,
                status="READY",
                target_rel_dir=decision.target_rel_dir,
                final_confidence=1.0,
                message=decision.reason,
            )

        # Ejecución real (no-dry-run)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target_path))
            self.console.print(
                f"[bold green][MOVIDO][/bold green] {path.parent.name}/{path.name} -> "
                f"[cyan]{target_path}[/cyan] ([magenta]{decision.reason}[/magenta])"
            )
            self.logger.info(
                "MOVED_NON_MEDIA | %s -> %s | reason=%s",
                path,
                target_path,
                decision.reason,
            )
        except Exception as exc:
            self.console.print(f"[bold red][ERROR AL MOVER][/bold red] {path.name}: {exc}")
            self.logger.error("MOVE_ERROR | %s -> %s | %s", path, target_path, exc)
            return ClassificationResult(
                source_path=path,
                status="ERROR",
                target_rel_dir=None,
                final_confidence=0.0,
                message=str(exc),
            )

        # Descompresión inteligente automática si es un archivo comprimido
        if decision.is_archive and config.AUTO_EXTRACT_ARCHIVES:
            ext_res = extract_archive_smart(target_path, descomprimidos_dir)
            if ext_res.ok and ext_res.extracted_dest is not None:
                self.console.print(
                    f"[bold green][DESCOMPRIMIDO][/bold green] {target_path.name} -> "
                    f"[cyan]{ext_res.extracted_dest}[/cyan] ({ext_res.message})"
                )
                self.logger.info(
                    "EXTRACTED | %s -> %s | wrapper_created=%s | %s",
                    target_path,
                    ext_res.extracted_dest,
                    ext_res.created_wrapper_folder,
                    ext_res.message,
                )
            else:
                self.console.print(
                    f"[yellow][AVISO DESCOMPRESIÓN][/yellow] {target_path.name}: {ext_res.message}"
                )
                self.logger.warning(
                    "EXTRACT_WARNING | %s | %s",
                    target_path,
                    ext_res.message,
                )

        return ClassificationResult(
            source_path=path,
            status="READY",
            target_rel_dir=decision.target_rel_dir,
            final_confidence=1.0,
            message=decision.reason,
        )

    def _is_duplicate_event(self, path: Path) -> bool:
        now = time.monotonic()
        expired = [k for k, (_, _, ts) in self._recent_events.items() if now - ts > 10.0]
        for k in expired:
            self._recent_events.pop(k, None)

        try:
            st = path.stat()
            sig = (st.st_mtime, st.st_size)
        except OSError:
            return False

        key = str(path.resolve())
        prev = self._recent_events.get(key)
        if prev and prev[0] == sig[0] and prev[1] == sig[1]:
            return True
        self._recent_events[key] = (sig[0], sig[1], now)
        return False

    def should_ignore(self, path: Path) -> bool:
        return should_ignore_path(path)

    def process_file(self, path: Path, wait_stability: bool = True) -> ClassificationResult | None:
        if self.should_ignore(path):
            return None

        if wait_stability and not wait_for_file_stability(path):
            self.logger.warning("SKIPPED_UNSTABLE | %s | El archivo desapareció o sigue bloqueado", path)
            return None

        if not path.exists() or not path.is_file():
            return None

        if self._is_duplicate_event(path):
            return None

        # Ruta rápida sin TensorFlow para Documentos, Programas, Cursores y Comprimidos
        if get_non_media_category(path) != "unsupported":
            return self._process_non_media(path)

        if self.classifier is None:
            return None

        result = self.classifier.classify_file(path)
        trace = format_decision_trace(result.decisions)

        if result.status == "NO_MODEL":
            self.console.print(f"[yellow][SIN MODELO][/yellow] {path.name} -> {result.message}")
            self.logger.warning("NO_MODEL | %s | %s", path, result.message)
            return result

        if result.status != "READY" or result.target_rel_dir is None:
            self.console.print(f"[red][ERROR][/red] {path.name} -> {result.message}")
            self.logger.error("ERROR | %s | %s", path, result.message)
            return result

        target_dir = self.dest_root / Path(result.target_rel_dir)
        target_path = resolve_non_colliding_path(target_dir, path.name)
        conf_pct = result.final_confidence * 100.0
        meets_threshold = result.final_confidence >= self.threshold

        if self.dry_run:
            verdict = "MOVERÍA" if meets_threshold else f"OMITIRÍA (< {self.threshold:.0%})"
            style = "bold cyan" if meets_threshold else "yellow"
            self.console.print(
                f"[{style}][DRY-RUN | {verdict}][/{style}] [white]{path.parent.name}/{path.name}[/white] "
                f"-> [green]{result.target_rel_dir}/[/green] "
                f"(Confianza: [bold]{conf_pct:.1f}%[/bold] | Ruta: {trace})"
            )
            self.logger.info(
                "DRY_RUN | %s | %s -> %s | conf=%.4f | threshold=%.2f | trace=%s",
                verdict,
                path,
                target_path,
                result.final_confidence,
                self.threshold,
                trace,
            )
            return result

        # Modo ejecución real (--no-dry-run / --execute)
        if not meets_threshold:
            self.console.print(
                f"[yellow][OMITIDO - BAJA CONFIANZA][/yellow] {path.parent.name}/{path.name} -> "
                f"Sugerido: {result.target_rel_dir}/ ({conf_pct:.1f}% < {self.threshold:.0%}) | {trace}"
            )
            self.logger.info(
                "SKIPPED_LOW_CONF | %s | suggested=%s | conf=%.4f < %.2f | trace=%s",
                path,
                target_path,
                result.final_confidence,
                self.threshold,
                trace,
            )
            return result

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target_path))
            self.console.print(
                f"[bold green][MOVIDO][/bold green] {path.parent.name}/{path.name} -> "
                f"[cyan]{target_path}[/cyan] (Confianza: [bold]{conf_pct:.1f}%[/bold] | {trace})"
            )
            self.logger.info(
                "MOVED | %s -> %s | conf=%.4f | trace=%s",
                path,
                target_path,
                result.final_confidence,
                trace,
            )
        except Exception as exc:
            self.console.print(f"[bold red][ERROR AL MOVER][/bold red] {path.name}: {exc}")
            self.logger.error("MOVE_ERROR | %s -> %s | %s", path, target_path, exc)

        return result


class OnDemandBatchDispatcher:
    """Despacha lotes de archivos a un subproceso efímero para liberar 100% de RAM de TensorFlow.

    El vigilante residente se mantiene en ~25 MB de RAM mientras no llegan archivos.
    Al detectar eventos en cualquiera de las carpetas vigiladas, agrupa los archivos durante
    `debounce_seconds` y ejecuta un único trabajador efímero (sin ventana de consola en Windows).
    """

    def __init__(
        self,
        dest_root: Path,
        threshold: float,
        dry_run: bool,
        num_frames: int,
        max_loaded_models: int,
        debounce_seconds: float,
        console: Console,
        logger: logging.Logger,
    ) -> None:
        self.dest_root = dest_root.resolve()
        self.threshold = threshold
        self.dry_run = dry_run
        self.num_frames = num_frames
        self.max_loaded_models = max_loaded_models
        self.debounce_seconds = max(0.5, debounce_seconds)
        self.console = console
        self.logger = logger

        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._non_media_lock = threading.Lock()
        self._pending: dict[str, Path] = {}
        self._timer: threading.Timer | None = None
        self._stopped = False
        self._non_media_service = MediaOrganizerService(
            classifier=None,
            dest_root=self.dest_root,
            threshold=self.threshold,
            dry_run=self.dry_run,
            console=self.console,
            logger=self.logger,
        )

    def _run_non_media_fast(self, path: Path) -> None:
        with self._non_media_lock:
            self._non_media_service.process_file(path, wait_stability=True)

    def submit(self, path: Path) -> None:
        if should_ignore_path(path):
            return
        resolved = path.resolve()
        # Documentos, programas y comprimidos se procesan al instante SIN cargar TensorFlow
        if get_non_media_category(resolved) != "unsupported":
            thread = threading.Thread(target=self._run_non_media_fast, args=(resolved,), daemon=True)
            thread.start()
            return
        with self._lock:
            if self._stopped:
                return
            self._pending[str(resolved)] = resolved
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._flush_async)
            self._timer.daemon = True
            self._timer.start()

    def _flush_async(self) -> None:
        thread = threading.Thread(target=self._run_worker_batch, daemon=True)
        thread.start()

    def _run_worker_batch(self) -> None:
        with self._worker_lock:
            with self._lock:
                if not self._pending:
                    return
                batch = list(self._pending.values())
                self._pending.clear()
                self._timer = None

            stable_paths: list[Path] = []
            for p in batch:
                if should_ignore_path(p):
                    continue
                if wait_for_file_stability(p):
                    stable_paths.append(p)
                else:
                    self.logger.warning("SKIPPED_UNSTABLE | %s | Archivo no estable en modo on-demand", p)

            if not stable_paths:
                return

            cmd = [
                sys.executable,
                "-m",
                "src.organizer_watcher",
                "--classify-files",
                *[str(p) for p in stable_paths],
                "--dest-root",
                str(self.dest_root),
                "--threshold",
                str(self.threshold),
                "--frames",
                str(self.num_frames),
                "--max-loaded-models",
                str(self.max_loaded_models),
                "--dry-run" if self.dry_run else "--no-dry-run",
            ]

            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.logger.info(
                "ON_DEMAND_DISPATCH | files=%d | mode=%s",
                len(stable_paths),
                "DRY_RUN" if self.dry_run else "EXECUTE",
            )
            self.console.print(
                f"[dim][ON-DEMAND] Lanzando clasificación efímera para {len(stable_paths)} archivo(s)...[/dim]"
            )
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(config.PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                    check=False,
                )
                if proc.stdout and sys.stdout is not None:
                    for line in proc.stdout.splitlines():
                        if line.strip():
                            self.console.print(line, markup=False)
                if proc.returncode != 0:
                    err_tail = (proc.stderr or "").strip()[-400:]
                    self.logger.error("ON_DEMAND_ERROR | rc=%d | %s", proc.returncode, err_tail)
            except Exception as exc:
                self.logger.error("ON_DEMAND_SPAWN_FAILED | %s", exc)

            # Si llegaron más archivos mientras corría el trabajador, re-programar automáticamente
            with self._lock:
                if self._pending and not self._stopped and self._timer is None:
                    self._timer = threading.Timer(self.debounce_seconds, self._flush_async)
                    self._timer.daemon = True
                    self._timer.start()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class MediaWatchdogHandler(FileSystemEventHandler):
    """Manejador de eventos de `watchdog` para creación y renombrado de descargas."""

    def __init__(
        self,
        service: MediaOrganizerService | None = None,
        dispatcher: OnDemandBatchDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.service = service
        self.dispatcher = dispatcher

    def _handle_path(self, raw_path: str) -> None:
        path = Path(raw_path)
        if self.dispatcher is not None:
            self.dispatcher.submit(path)
        elif self.service is not None:
            self.service.process_file(path, wait_stability=True)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_path(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", None)
        if dest:
            self._handle_path(dest)


def validate_safe_directories(watch_dirs: Path | Sequence[Path], dest_root: Path) -> list[Path]:
    """Valida una o varias carpetas vigiladas evitando bucles infinitos o vigilar la raíz."""
    items = [watch_dirs] if isinstance(watch_dirs, Path) else list(watch_dirs)
    d = dest_root.resolve()
    proj = config.PROJECT_ROOT.resolve()

    validated: list[Path] = []
    seen: set[str] = set()

    for raw_w in items:
        w = raw_w.resolve()
        if w == proj:
            raise ValueError(
                f"Por seguridad no se permite vigilar la raíz del proyecto ({proj}). "
                f"Usa carpetas dedicadas como '{config.INBOX_DIR}' o `--profile pc`."
            )
        if w == d or d in w.parents:
            raise ValueError(
                f"La carpeta vigilada ({w}) no puede ser igual ni estar dentro de la carpeta destino ({d})."
            )
        key = str(w).lower()
        if key not in seen:
            seen.add(key)
            validated.append(w)

    if not validated:
        raise ValueError("Debes especificar al menos una carpeta a vigilar.")
    return validated


def resolve_watch_directories(args: argparse.Namespace) -> list[Path]:
    """Resuelve la lista final de carpetas a vigilar según `--watch-dir` o `--profile`."""
    if args.watch_dir:
        return [Path(p) for p in args.watch_dir]
    if args.profile == "pc":
        return list(config.DEFAULT_PC_WATCH_DIRS)
    return [config.INBOX_DIR]


def print_startup_banner(
    console: Console,
    watch_dirs: list[Path],
    dest_root: Path,
    dry_run: bool,
    on_demand: bool,
    threshold: float,
    classifier: HierarchicalClassifier,
    log_path: Path,
) -> None:
    mode_badge = (
        "[bold cyan]DRY-RUN (MODO SEGURO - NO MUEVE ARCHIVOS)[/bold cyan]"
        if dry_run
        else "[bold red]EJECUCIÓN REAL (MOVERÁ ARCHIVOS >= UMBRAL)[/bold red]"
    )
    ram_badge = (
        "[bold green]ON-DEMAND (Bajo consumo RAM ~25 MB en reposo)[/bold green]"
        if on_demand
        else "[yellow]RESIDENTE (TensorFlow en memoria)[/yellow]"
    )
    dirs_formatted = "\n".join(f"  - [cyan]{w}[/cyan]" for w in watch_dirs)

    console.print()
    console.print(
        Panel(
            f"[bold white]MOTOR DE CLASIFICACIÓN Y VIGILANCIA MULTIMEDIA (MULTI-CARPETA)[/bold white]\n"
            f"Modo:   {mode_badge}\n"
            f"RAM:    {ram_badge}\n"
            f"Carpetas vigiladas ({len(watch_dirs)}):\n{dirs_formatted}\n"
            f"Raíz destino:     [cyan]{dest_root}[/cyan]\n"
            f"Umbral confianza: [yellow]{threshold:.0%}[/yellow]  |  "
            f"Frames/video: [yellow]{classifier.num_frames}[/yellow]  |  "
            f"Log: [dim]{log_path}[/dim]",
            border_style="cyan" if dry_run else "red",
        )
    )

    tbl = Table(title="Modelos Promovidos en Producción Detectados", box=box.ROUNDED)
    tbl.add_column("Nodo", style="cyan")
    tbl.add_column("Nivel", justify="center")
    tbl.add_column("Preprocesado", style="magenta")
    tbl.add_column("Clases", style="white")
    tbl.add_column("Val / Test Acc", justify="right", style="green")

    if classifier.promoted_models:
        outdated_resize: list[str] = []
        for node_path, info in sorted(classifier.promoted_models.items()):
            cmap_path = classifier.models_dir / info.get("class_map_file", "")
            resize_mode = "stretch (anterior)"
            if cmap_path.exists():
                try:
                    with cmap_path.open("r", encoding="utf-8") as fh:
                        resize_mode = json.load(fh).get("resize_mode", "stretch (anterior)")
                except Exception:
                    pass
            if resize_mode != config.RESIZE_MODE:
                outdated_resize.append(info.get("model_id", node_path))

            tbl.add_row(
                f"{node_path}/",
                str(info.get("level", "?")),
                resize_mode,
                ", ".join(info.get("class_names", [])),
                f"{info.get('val_accuracy', 0.0):.1%} / {info.get('test_accuracy', 0.0):.1%}",
            )
        console.print(tbl)
        if outdated_resize:
            console.print(
                f"[bold yellow][!] Aviso de preprocesamiento ({config.RESIZE_MODE}):[/bold yellow] "
                f"Los siguientes modelos fueron entrenados antes de activar `{config.RESIZE_MODE}` "
                f"y conviene reentrenarlos: [cyan]{', '.join(outdated_resize)}[/cyan]"
            )
    else:
        console.print(
            "[yellow][!] Aún no hay modelos promovidos en `models/production/`.[/yellow]\n"
            "    Entrena y aprueba modelos primero con: [bold cyan]python -m src.train_supervised[/bold cyan]"
        )

    console.print(f"[dim]Rutas deterministas multimedia: {classifier.deterministic_routes}[/dim]")
    console.print(
        "[dim]Reglas sin TensorFlow (<1s): "
        f"Programas (.exe/.msi) -> {config.DEST_PROGRAMAS_REL}/ | "
        f"Comprimidos (.zip/.rar/.7z) -> {config.DEST_COMPRIMIDOS_REL}/ + auto-extracción inteligente en {config.DEST_DESCOMPRIMIDOS_REL}/ | "
        f"Documentos (.pdf/.docx/.ani/.cur) -> documentos/{{PDFs, WordS, Universidad, Mi_Cursor}}/[/dim]\n"
    )


def _write_pid_file(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pid_file(pid_path: Path) -> None:
    try:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vigilante multi-carpeta de archivos multimedia con votación y modo Dry-Run."
    )
    parser.add_argument(
        "--profile",
        choices=["inbox", "pc"],
        default="inbox",
        help="Perfil de carpetas si no se usa `--watch-dir`: `inbox` (pruebas) o `pc` (Screenshots + Downloads).",
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        action="append",
        default=None,
        help="Carpeta a vigilar (puede repetirse varias veces para vigilar múltiples carpetas).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=config.DATASET_DIR,
        help=f"Carpeta raíz con la estructura destino (por defecto: {config.DATASET_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=config.DRY_RUN_DEFAULT,
        help="Modo seguro activado por defecto (usa `--no-dry-run` para habilitar movimientos reales).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Atajo explícito equivalente a `--no-dry-run` (mueve archivos si superan el umbral).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confianza mínima entre 0 y 1 para mover archivos (por defecto: {config.CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=config.VIDEO_FRAMES_PER_FILE,
        choices=[3, 4, 5],
        help=f"Fotogramas extraídos por video/animación para votación (por defecto: {config.VIDEO_FRAMES_PER_FILE}).",
    )
    parser.add_argument(
        "--max-loaded-models",
        type=int,
        default=2,
        help="Máximo de modelos .keras simultáneos en RAM (por defecto: 2, ideal para 3.44 GB RAM).",
    )
    parser.add_argument(
        "--on-demand",
        action="store_true",
        help="Modo bajo consumo RAM (~25 MB en reposo): carga TensorFlow en un subproceso efímero solo cuando llegan archivos.",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=config.ON_DEMAND_DEBOUNCE_SECONDS,
        help=f"Segundos para agrupar archivos seguidos en modo `--on-demand` (por defecto: {config.ON_DEMAND_DEBOUNCE_SECONDS}).",
    )
    parser.add_argument(
        "--scan-existing",
        action="store_true",
        help="Procesa también los archivos ya presentes en todas las carpetas vigiladas al iniciar.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Procesa una sola vez los archivos actuales en las carpetas vigiladas y finaliza.",
    )
    parser.add_argument(
        "--classify-files",
        nargs="+",
        type=Path,
        default=None,
        help="Uso interno / puntual: clasifica una lista exacta de archivos y termina (libera RAM al salir).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = make_safe_console()

    dry_run = False if args.execute else bool(args.dry_run)
    if not (0.0 < args.threshold <= 1.0):
        console.print("[bold red]ERROR:[/bold red] `--threshold` debe estar entre 0.01 y 1.0.")
        return 1

    dest_root = args.dest_root.resolve()
    log_path = config.LOGS_DIR / "organizer.log"
    logger = setup_logger(log_path)

    # Modo trabajador efímero: clasifica lista exacta de archivos y sale liberando RAM
    if args.classify_files:
        classifier = HierarchicalClassifier(
            manifest_path=config.HIERARCHY_MANIFEST_PATH,
            models_dir=config.MODELS_PROD_DIR,
            num_frames=args.frames,
            max_loaded_models=args.max_loaded_models,
        )
        service = MediaOrganizerService(
            classifier=classifier,
            dest_root=dest_root,
            threshold=args.threshold,
            dry_run=dry_run,
            console=console,
            logger=logger,
        )
        for fpath in args.classify_files:
            service.process_file(fpath.resolve(), wait_stability=False)
        return 0

    raw_watch_dirs = resolve_watch_directories(args)
    for w in raw_watch_dirs:
        w.mkdir(parents=True, exist_ok=True)

    try:
        watch_dirs = validate_safe_directories(raw_watch_dirs, dest_root)
    except ValueError as exc:
        console.print(f"[bold red]ERROR DE SEGURIDAD:[/bold red] {exc}")
        return 1

    classifier = HierarchicalClassifier(
        manifest_path=config.HIERARCHY_MANIFEST_PATH,
        models_dir=config.MODELS_PROD_DIR,
        num_frames=args.frames,
        max_loaded_models=args.max_loaded_models,
    )

    print_startup_banner(
        console=console,
        watch_dirs=watch_dirs,
        dest_root=dest_root,
        dry_run=dry_run,
        on_demand=bool(args.on_demand and not args.once),
        threshold=args.threshold,
        classifier=classifier,
        log_path=log_path,
    )

    logger.info(
        "STARTUP | mode=%s | ram=%s | watch=%s | dest=%s | threshold=%.2f | models=%d",
        "DRY_RUN" if dry_run else "EXECUTE",
        "ON_DEMAND" if (args.on_demand and not args.once) else "RESIDENT",
        [str(w) for w in watch_dirs],
        dest_root,
        args.threshold,
        len(classifier.promoted_models),
    )

    # Escaneo inicial si se solicita --scan-existing o --once
    if args.scan_existing or args.once:
        service = MediaOrganizerService(
            classifier=classifier,
            dest_root=dest_root,
            threshold=args.threshold,
            dry_run=dry_run,
            console=console,
            logger=logger,
        )
        total_found = 0
        for wdir in watch_dirs:
            existing_files = sorted(
                p for p in wdir.iterdir() if p.is_file() and not should_ignore_path(p)
            )
            if not existing_files:
                console.print(f"[dim]Sin archivos multimedia pendientes en {wdir}[/dim]")
                continue
            total_found += len(existing_files)
            console.print(f"[cyan]Analizando {len(existing_files)} archivo(s) en {wdir}...[/cyan]")
            for fpath in existing_files:
                service.process_file(fpath, wait_stability=False)

        if args.once:
            console.print(f"[green]Ejecución única (`--once`) finalizada ({total_found} archivo(s) analizados).[/green]")
            return 0
        # En modo on-demand liberamos el servicio/modelos antes de quedar residentes
        if args.on_demand:
            del service
            classifier._model_cache.clear()
            gc.collect()

    # Vigilancia continua multi-carpeta con watchdog
    dispatcher: OnDemandBatchDispatcher | None = None
    resident_service: MediaOrganizerService | None = None

    if args.on_demand:
        dispatcher = OnDemandBatchDispatcher(
            dest_root=dest_root,
            threshold=args.threshold,
            dry_run=dry_run,
            num_frames=args.frames,
            max_loaded_models=args.max_loaded_models,
            debounce_seconds=args.debounce,
            console=console,
            logger=logger,
        )
        handler = MediaWatchdogHandler(dispatcher=dispatcher)
    else:
        resident_service = MediaOrganizerService(
            classifier=classifier,
            dest_root=dest_root,
            threshold=args.threshold,
            dry_run=dry_run,
            console=console,
            logger=logger,
        )
        handler = MediaWatchdogHandler(service=resident_service)

    observer = Observer()
    for wdir in watch_dirs:
        observer.schedule(handler, str(wdir), recursive=False)

    _write_pid_file(config.WATCHER_PID_FILE)
    observer.start()

    console.print(
        f"[bold green]Vigilancia activa en {len(watch_dirs)} carpeta(s).[/bold green] "
        "Presiona [bold]Ctrl+C[/bold] para detener.\n"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[yellow]Deteniendo vigilante...[/yellow]")
    finally:
        if dispatcher is not None:
            dispatcher.stop()
        observer.stop()
        observer.join()
        _remove_pid_file(config.WATCHER_PID_FILE)
        logger.info("SHUTDOWN | Vigilante detenido.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
