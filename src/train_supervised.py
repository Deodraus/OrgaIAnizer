"""Paso 3: Pipeline de Entrenamiento Supervisado, Interactivo y Visual (Jerárquico).

Principios de seguridad y supervisión:
1. NUNCA entrena sin mostrar antes el plan de datos/splits y recibir tu confirmación explícita.
2. Configura TensorBoard en vivo + guarda gráficos de Loss/Accuracy y Matriz de Confusión.
3. Guarda los pesos primero en `models/staging/` y SOLO promueve a `models/production/`
   tras mostrarte las métricas de validación/test y recibir tu aprobación explícita.
4. Optimizado para 3.44 GB de RAM usable en CPU (caché en disco temporal, batch_size=8,
   MobileNetV2 con Transfer Learning en 2 fases y limpieza de sesión entre modelos).

Uso:
    python -m src.train_supervised
    python -m src.train_supervised --model imagenes
    python -m src.train_supervised --model imagenes__capturas_pantalla --max-cap 250
    python -m src.train_supervised --plan-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.media_extractor import extract_frames, get_media_type

# Reduce verbosidad de logs C++ de TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class SampleRecord:
    """Representa un archivo multimedia individual y su clase en un nodo jerárquico."""

    path: Path
    class_index: int
    class_name: str
    media_type: str


@dataclass
class NodeSpec:
    """Especificación de un clasificador jerárquico entrenable."""

    model_id: str
    node_path: str
    level: int
    class_names: list[str]
    files_by_class: dict[str, list[Path]]


def collect_media_files_recursive(folder: Path) -> list[Path]:
    """Recopila archivos multimedia soportados en una carpeta y sus subcarpetas."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort(key=lambda s: s.lower())
        filenames.sort(key=lambda s: s.lower())
        cur = Path(dirpath)
        for fname in filenames:
            if fname.lower() in config.IGNORED_SYSTEM_FILES:
                continue
            fpath = cur / fname
            if get_media_type(fpath) != "unsupported":
                files.append(fpath)
    return files


def discover_trainable_nodes(dataset_dir: Path) -> list[NodeSpec]:
    """Descubre todos los nodos jerárquicos con >= 2 clases pobladas dentro de imagenes/ y videos/."""
    nodes: list[NodeSpec] = []

    for branch in config.MULTIMEDIA_BRANCHES:
        branch_dir = dataset_dir / branch
        if not branch_dir.exists() or not branch_dir.is_dir():
            continue

        for dirpath, dirnames, _ in os.walk(branch_dir):
            dirnames.sort(key=lambda s: s.lower())
            cur_dir = Path(dirpath)
            rel_path = cur_dir.relative_to(dataset_dir).as_posix()

            # Si es una ruta determinista configurada (ej. imagenes/galeria -> fotos), omitir modelo
            if rel_path in config.DETERMINISTIC_ROUTES:
                continue

            child_dirs = sorted(
                [cur_dir / d for d in dirnames if (cur_dir / d).is_dir()],
                key=lambda p: p.name.lower(),
            )
            if len(child_dirs) < 2:
                continue

            files_by_class: dict[str, list[Path]] = {}
            for child in child_dirs:
                media_files = collect_media_files_recursive(child)
                if media_files:
                    files_by_class[child.name] = media_files

            if len(files_by_class) >= 2:
                class_names = sorted(files_by_class.keys(), key=lambda s: s.lower())
                depth = len(Path(rel_path).parts)
                nodes.append(
                    NodeSpec(
                        model_id=rel_path.replace("/", "__"),
                        node_path=rel_path,
                        level=depth + 1,
                        class_names=class_names,
                        files_by_class={c: files_by_class[c] for c in class_names},
                    )
                )

    nodes.sort(key=lambda n: (n.level, n.node_path))
    return nodes


def build_stratified_file_splits(
    node: NodeSpec,
    max_cap: int,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord], list[str]]:
    """Divide los archivos en Train (70%), Val (15%) y Test (15%) estratificado por clase.

    La división siempre se realiza a NIVEL DE ARCHIVO completo para evitar fuga de datos
    entre fotogramas del mismo video/GIF. Para clases pequeñas (< 7 archivos), aplica un
    split adaptativo garantizando mínimo 1 archivo en val y 1 en test si hay >= 3 archivos.
    """
    rng = random.Random(seed)
    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []
    test_records: list[SampleRecord] = []
    warnings_list: list[str] = []

    for cls_idx, cls_name in enumerate(node.class_names):
        all_files = list(node.files_by_class[cls_name])
        rng.shuffle(all_files)

        original_n = len(all_files)
        if original_n > max_cap:
            all_files = all_files[:max_cap]

        n = len(all_files)
        if n < 3:
            raise ValueError(
                f"La clase '{cls_name}' en '{node.node_path}' solo tiene {n} archivo(s). "
                "Se requieren al menos 3 archivos por clase para entrenar, validar y probar."
            )

        if n < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT:
            n_val = 1
            n_test = 1
            n_train = n - 2
            warnings_list.append(
                f"Clase '{cls_name}' tiene solo {n} archivos (<7): se aplicó split adaptativo "
                f"{n_train}/{n_val}/{n_test} (métricas de test orientativas)."
            )
        else:
            n_val = max(1, int(round(n * config.VAL_RATIO)))
            n_test = max(1, int(round(n * config.TEST_RATIO)))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                remaining = n - 1
                n_val = max(1, remaining // 2)
                n_test = max(1, remaining - n_val)

        if original_n > max_cap:
            warnings_list.append(
                f"Clase '{cls_name}' limitada de {original_n} a {max_cap} archivos "
                "para reducir desbalanceo y tiempo en CPU."
            )

        val_slice = all_files[:n_val]
        test_slice = all_files[n_val : n_val + n_test]
        train_slice = all_files[n_val + n_test :]

        for f in train_slice:
            train_records.append(SampleRecord(f, cls_idx, cls_name, get_media_type(f)))
        for f in val_slice:
            val_records.append(SampleRecord(f, cls_idx, cls_name, get_media_type(f)))
        for f in test_slice:
            test_records.append(SampleRecord(f, cls_idx, cls_name, get_media_type(f)))

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records, warnings_list


def compute_balanced_class_weights(
    train_records: list[SampleRecord],
    num_classes: int,
    max_weight: float = 8.0,
) -> dict[int, float]:
    """Calcula pesos balanceados por clase (acotados para estabilidad numérica)."""
    y_train = np.array([r.class_index for r in train_records], dtype=np.int32)
    classes = np.arange(num_classes, dtype=np.int32)
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): round(float(min(w, max_weight)), 4) for c, w in zip(classes, raw_weights)}


def print_node_plan(
    console: Console,
    node: NodeSpec,
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    class_weights: dict[int, float],
    warnings_list: list[str],
    args: argparse.Namespace,
) -> None:
    """Muestra la tabla detallada del plan de entrenamiento antes de pedir confirmación."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]MODELO JERÁRQUICO:[/bold cyan] [bold white]{node.model_id}[/bold white] "
            f"(Carpeta: {node.node_path}/ | Predice Nivel {node.level})\n"
            f"Backbone: [green]{config.BACKBONE_NAME} (alpha={args.alpha})[/green]  |  "
            f"Batch: [yellow]{args.batch_size}[/yellow]  |  "
            f"Épocas Fase 1: [yellow]{args.epochs}[/yellow]  |  "
            f"Frames/video: [yellow]{args.frames}[/yellow]  |  "
            f"Tope/clase: [yellow]{args.max_cap}[/yellow]",
            border_style="cyan",
        )
    )

    train_c = Counter(r.class_name for r in train_recs)
    val_c = Counter(r.class_name for r in val_recs)
    test_c = Counter(r.class_name for r in test_recs)

    tbl = Table(title="Distribución Estratificada por Archivo (Sin Fuga entre Splits)", box=box.ROUNDED)
    tbl.add_column("ID", justify="right", style="dim")
    tbl.add_column("Clase (Subcarpeta)", style="cyan")
    tbl.add_column("Original", justify="right")
    tbl.add_column("Train (70%)", justify="right", style="green")
    tbl.add_column("Val (15%)", justify="right", style="yellow")
    tbl.add_column("Test (15%)", justify="right", style="magenta")
    tbl.add_column("class_weight", justify="right", style="bold")

    for idx, cls_name in enumerate(node.class_names):
        tbl.add_row(
            str(idx),
            cls_name,
            str(len(node.files_by_class[cls_name])),
            str(train_c[cls_name]),
            str(val_c[cls_name]),
            str(test_c[cls_name]),
            f"{class_weights[idx]:.2f}",
        )
    tbl.add_section()
    tbl.add_row(
        "-",
        "[bold]TOTAL ARCHIVOS[/bold]",
        str(sum(len(v) for v in node.files_by_class.values())),
        f"[bold]{len(train_recs)}[/bold]",
        f"[bold]{len(val_recs)}[/bold]",
        f"[bold]{len(test_recs)}[/bold]",
        "-",
    )
    console.print(tbl)

    for w in warnings_list:
        console.print(f"  [yellow][!][/yellow] {w}")
    console.print()


def _build_tf_dataset(
    records: list[SampleRecord],
    num_frames: int,
    batch_size: int,
    cache_file: Path,
    shuffle: bool,
    seed: int,
):
    """Construye un tf.data.Dataset con caché en disco temporal (muy bajo consumo de RAM)."""
    import tensorflow as tf

    file_items = [(str(r.path), int(r.class_index)) for r in records]

    def frame_generator():
        for fpath_str, label in file_items:
            try:
                frames = extract_frames(fpath_str, num_frames=num_frames, apply_preprocess=True)
            except Exception:
                continue
            for i in range(frames.shape[0]):
                yield frames[i], label

    output_signature = (
        tf.TensorSpec(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    ds = tf.data.Dataset.from_generator(frame_generator, output_signature=output_signature)
    ds = ds.cache(str(cache_file))
    if shuffle:
        ds = ds.shuffle(buffer_size=48, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def build_mobilenetv2_classifier(
    num_classes: int,
    alpha: float,
    learning_rate: float,
):
    """Crea el modelo MobileNetV2 con Data Augmentation ligero en [-1, 1] y backbone congelado."""
    import keras
    from keras import layers

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.06, fill_mode="constant", fill_value=0.0),
            layers.RandomBrightness(factor=0.12, value_range=(-1.0, 1.0)),
            layers.RandomContrast(factor=0.10, value_range=(-1.0, 1.0)),
        ],
        name="light_augmentation",
    )

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3),
        alpha=alpha,
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), name="media_input")
    x = data_augmentation(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_hierarchical")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def evaluate_files_by_voting(
    model,
    records: list[SampleRecord],
    num_frames: int,
    console: Console | None = None,
    split_name: str = "Evaluación",
    batch_size: int = config.BATCH_SIZE,
) -> dict[str, Any]:
    """Evalúa a nivel de archivo completo promediando probabilidades entre fotogramas (votación).

    Usa inferencia directa por lotes (`model(x, training=False)`) para evitar el overhead
    de llamar `model.predict()` archivo por archivo en CPU.
    """
    import tensorflow as tf
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    errors: list[dict[str, Any]] = []

    pending_frames: list[np.ndarray] = []
    pending_meta: list[tuple[SampleRecord, int]] = []  # (record, num_frames_extracted)

    def _flush_batch() -> None:
        if not pending_frames:
            return
        batch_arr = np.concatenate(pending_frames, axis=0)
        probs_all = model(tf.convert_to_tensor(batch_arr, dtype=tf.float32), training=False).numpy()

        offset = 0
        for rec, n_f in pending_meta:
            file_probs = probs_all[offset : offset + n_f]
            offset += n_f
            mean_probs = np.mean(file_probs, axis=0)
            pred_idx = int(np.argmax(mean_probs))
            conf = float(mean_probs[pred_idx])

            y_true.append(rec.class_index)
            y_pred.append(pred_idx)
            confidences.append(conf)

            if pred_idx != rec.class_index:
                errors.append(
                    {
                        "file": rec.path.name,
                        "path": str(rec.path),
                        "true_idx": rec.class_index,
                        "pred_idx": pred_idx,
                        "confidence": round(conf, 4),
                    }
                )

        pending_frames.clear()
        pending_meta.clear()

    total_recs = max(len(records), 1)
    if console is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{split_name}...[/cyan]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
    else:
        progress = None

    if progress is not None:
        progress.start()
        task_id = progress.add_task(split_name, total=total_recs)
    else:
        task_id = None

    try:
        buffered_frames_count = 0
        for rec in records:
            if progress is not None and task_id is not None:
                progress.advance(task_id, 1)
            try:
                frames = extract_frames(rec.path, num_frames=num_frames, apply_preprocess=True)
            except Exception:
                continue

            pending_frames.append(frames)
            pending_meta.append((rec, int(frames.shape[0])))
            buffered_frames_count += int(frames.shape[0])

            if buffered_frames_count >= batch_size:
                _flush_batch()
                buffered_frames_count = 0

        _flush_batch()
    finally:
        if progress is not None:
            progress.stop()

    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "y_true": [], "y_pred": [], "errors": []}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float(np.mean(y_true_arr == y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "mean_confidence": float(np.mean(confidences)),
        "y_true": y_true,
        "y_pred": y_pred,
        "errors": errors,
    }


def save_training_curves(
    history_dict: dict[str, list[float]],
    model_id: str,
    phase1_end_epoch: int | None,
    output_path: Path,
) -> None:
    """Genera y guarda las curvas de Loss y Accuracy (entrenamiento vs. validación)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_range = list(range(1, len(history_dict.get("loss", [])) + 1))
    if not epochs_range:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # Loss
    ax1.plot(epochs_range, history_dict["loss"], marker="o", label="Train Loss")
    if "val_loss" in history_dict:
        ax1.plot(epochs_range, history_dict["val_loss"], marker="s", label="Val Loss")
    if phase1_end_epoch is not None and phase1_end_epoch < len(epochs_range):
        ax1.axvline(phase1_end_epoch + 0.5, color="gray", linestyle="--", label="Inicio Fine-Tuning")
    ax1.set_title(f"{model_id} — Evolución de Loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Accuracy
    ax2.plot(epochs_range, history_dict["accuracy"], marker="o", label="Train Accuracy")
    if "val_accuracy" in history_dict:
        ax2.plot(epochs_range, history_dict["val_accuracy"], marker="s", label="Val Accuracy")
    if phase1_end_epoch is not None and phase1_end_epoch < len(epochs_range):
        ax2.axvline(phase1_end_epoch + 0.5, color="gray", linestyle="--", label="Inicio Fine-Tuning")
    ax2.set_title(f"{model_id} — Evolución de Accuracy")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def save_confusion_matrix_plot(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    model_id: str,
    split_label: str,
    output_path: Path,
) -> np.ndarray:
    """Genera y guarda el mapa de calor de la Matriz de Confusión (conteo y % por fila)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), np.where(row_sums == 0, 1, row_sums)) * 100.0

    annot = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot[r, c] = f"{cm[r, c]}\n({cm_pct[r, c]:.0f}%)"

    fig_size = max(5.5, 1.6 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_size + 1.2, fig_size), constrained_layout=True)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusión ({split_label}) — {model_id}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Carpeta Real")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return cm


def update_hierarchy_manifest(
    node: NodeSpec,
    prod_model_path: Path,
    prod_class_map_path: Path,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    """Actualiza models/production/hierarchy_manifest.json tras la aprobación explícita del usuario."""
    manifest_path = config.HIERARCHY_MANIFEST_PATH
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        manifest = {
            "schema_version": 1,
            "level1_media_routing": {
                "image": "imagenes",
                "video": "videos",
            },
            "deterministic_routes": dict(config.DETERMINISTIC_ROUTES),
            "models": {},
        }

    manifest["deterministic_routes"] = dict(config.DETERMINISTIC_ROUTES)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["models"][node.node_path] = {
        "model_id": node.model_id,
        "level": node.level,
        "model_file": prod_model_path.name,
        "class_map_file": prod_class_map_path.name,
        "class_names": node.class_names,
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def _cleanup_cache_files(cache_prefix: Path) -> None:
    """Elimina archivos temporales de tf.data.Dataset.cache() en disco."""
    if not cache_prefix.parent.exists():
        return
    for item in cache_prefix.parent.glob(f"{cache_prefix.name}*"):
        try:
            item.unlink()
        except OSError:
            pass


def ask_yes_no(console: Console, question: str, default_no: bool = True) -> bool:
    """Solicita confirmación explícita al usuario (por defecto NO por seguridad)."""
    suffix = " [s/N]: " if default_no else " [S/n]: "
    answer = console.input(f"[bold yellow]{question}{suffix}[/bold yellow]").strip().lower()
    if not answer:
        return not default_no
    return answer in {"s", "si", "sí", "y", "yes"}


def train_single_node(
    node: NodeSpec,
    args: argparse.Namespace,
    console: Console,
) -> bool:
    """Ejecuta el flujo interactivo supervisado completo para un nodo jerárquico."""
    train_recs, val_recs, test_recs, warnings_list = build_stratified_file_splits(
        node=node,
        max_cap=args.max_cap,
        seed=config.RANDOM_SEED,
    )
    class_weights = compute_balanced_class_weights(train_recs, num_classes=len(node.class_names))

    print_node_plan(
        console=console,
        node=node,
        train_recs=train_recs,
        val_recs=val_recs,
        test_recs=test_recs,
        class_weights=class_weights,
        warnings_list=warnings_list,
        args=args,
    )

    if args.plan_only:
        console.print("[green][PLAN-ONLY][/green] Verificación de splits completada sin entrenar.\n")
        return True

    if not sys.stdin.isatty():
        console.print(
            "[bold red]BLOQUEADO:[/bold red] El entrenamiento supervisado requiere una terminal "
            "interactiva para confirmar cada paso. No se ejecutará en segundo plano."
        )
        return False

    if not ask_yes_no(console, f"¿Confirmas iniciar el entrenamiento del modelo '{node.model_id}'?"):
        console.print(f"[yellow]Entrenamiento cancelado por el usuario para '{node.model_id}'.[/yellow]\n")
        return False

    # Import tardío de TensorFlow/Keras solo tras la confirmación explícita del usuario
    console.print("[cyan]Cargando TensorFlow/Keras y construyendo MobileNetV2...[/cyan]")
    import keras
    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = config.TENSORBOARD_DIR / f"{node.model_id}_{run_stamp}"
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            "[bold green]MONITOREO EN VIVO CON TENSORBOARD[/bold green]\n"
            "Abre otra terminal en esta carpeta y ejecuta:\n"
            f"  [bold cyan]python -m tensorboard.main --logdir \"{config.TENSORBOARD_DIR}\"[/bold cyan]\n"
            "Luego abre en tu navegador: [underline]http://localhost:6006[/underline]",
            border_style="green",
        )
    )

    cache_dir = config.LOGS_DIR / "tmp_cache"
    train_cache = cache_dir / f"{node.model_id}_{run_stamp}_train.cache"
    val_cache = cache_dir / f"{node.model_id}_{run_stamp}_val.cache"

    model = None
    try:
        train_ds = _build_tf_dataset(
            records=train_recs,
            num_frames=args.frames,
            batch_size=args.batch_size,
            cache_file=train_cache,
            shuffle=True,
            seed=config.RANDOM_SEED,
        )
        val_ds = _build_tf_dataset(
            records=val_recs,
            num_frames=args.frames,
            batch_size=args.batch_size,
            cache_file=val_cache,
            shuffle=False,
            seed=config.RANDOM_SEED,
        )

        model, backbone = build_mobilenetv2_classifier(
            num_classes=len(node.class_names),
            alpha=args.alpha,
            learning_rate=config.LEARNING_RATE_PHASE1,
        )

        callbacks_p1 = [
            keras.callbacks.TensorBoard(
                log_dir=str(tb_log_dir),
                histogram_freq=0,
                write_graph=False,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        console.print(f"\n[bold cyan]=== FASE 1: Transfer Learning (backbone congelado, max {args.epochs} épocas) ===[/bold cyan]")
        hist1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks_p1,
            verbose=1,
        )

        combined_history: dict[str, list[float]] = {
            k: [float(v) for v in vals] for k, vals in hist1.history.items()
        }
        phase1_epochs_ran = len(combined_history.get("loss", []))

        best_val_acc = max(combined_history.get("val_accuracy", [0.0]))
        best_val_loss = min(combined_history.get("val_loss", [float("inf")]))
        console.print(
            f"\n[bold]Resumen Fase 1:[/bold] Mejor Val Accuracy = [green]{best_val_acc:.2%}[/green] | "
            f"Mejor Val Loss = [yellow]{best_val_loss:.4f}[/yellow]"
        )

        # Fase 2 opcional supervisada (Fine-Tuning ligero)
        if ask_yes_no(
            console,
            f"¿Deseas ejecutar Fase 2 (Fine-Tuning de las últimas {config.FINETUNE_UNFREEZE_LAYERS} "
            f"capas, lr={config.LEARNING_RATE_PHASE2}, max {config.EPOCHS_PHASE2_FINETUNE} épocas)?",
        ):
            backbone.trainable = True
            for layer in backbone.layers[: -config.FINETUNE_UNFREEZE_LAYERS]:
                layer.trainable = False

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_PHASE2),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            callbacks_p2 = [
                keras.callbacks.TensorBoard(
                    log_dir=str(tb_log_dir),
                    histogram_freq=0,
                    write_graph=False,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
            console.print("\n[bold cyan]=== FASE 2: Fine-Tuning Ligero ===[/bold cyan]")
            hist2 = model.fit(
                train_ds,
                validation_data=val_ds,
                initial_epoch=phase1_epochs_ran,
                epochs=phase1_epochs_ran + config.EPOCHS_PHASE2_FINETUNE,
                class_weight=class_weights,
                callbacks=callbacks_p2,
                verbose=1,
            )
            for k, vals in hist2.history.items():
                combined_history.setdefault(k, []).extend(float(v) for v in vals)

        # Evaluación completa por votación de fotogramas en Validación y Test
        console.print("\n[cyan]Evaluando por archivo completo (votación por fotogramas) en Validación y Test...[/cyan]")
        val_eval = evaluate_files_by_voting(
            model,
            val_recs,
            num_frames=args.frames,
            console=console,
            split_name="Evaluando Validación",
            batch_size=args.batch_size,
        )
        test_eval = evaluate_files_by_voting(
            model,
            test_recs,
            num_frames=args.frames,
            console=console,
            split_name="Evaluando Test",
            batch_size=args.batch_size,
        )

        # Guardar gráficos (curvas + matriz de confusión)
        curves_png = config.EVALUATIONS_DIR / f"{node.model_id}_curves.png"
        cm_png = config.EVALUATIONS_DIR / f"{node.model_id}_confusion_matrix.png"
        metrics_json = config.EVALUATIONS_DIR / f"{node.model_id}_metrics.json"

        save_training_curves(
            history_dict=combined_history,
            model_id=node.model_id,
            phase1_end_epoch=phase1_epochs_ran,
            output_path=curves_png,
        )
        save_confusion_matrix_plot(
            y_true=test_eval["y_true"],
            y_pred=test_eval["y_pred"],
            class_names=node.class_names,
            model_id=node.model_id,
            split_label="Test",
            output_path=cm_png,
        )

        # Reporte de clasificación detallado por subcarpeta
        cls_report_text = classification_report(
            test_eval["y_true"],
            test_eval["y_pred"],
            labels=list(range(len(node.class_names))),
            target_names=node.class_names,
            zero_division=0,
        )

        console.print()
        console.print(
            Panel(
                f"[bold]RESULTADOS DE EVALUACIÓN POR ARCHIVO — {node.model_id}[/bold]\n"
                f"Validación -> Accuracy: [green]{val_eval['accuracy']:.2%}[/green] | "
                f"Macro-F1: [green]{val_eval['macro_f1']:.2%}[/green]\n"
                f"Test       -> Accuracy: [bold cyan]{test_eval['accuracy']:.2%}[/bold cyan] | "
                f"Macro-F1: [bold cyan]{test_eval['macro_f1']:.2%}[/bold cyan] | "
                f"Confianza media: {test_eval.get('mean_confidence', 0.0):.2%}\n\n"
                f"Curvas guardadas: [cyan]{curves_png}[/cyan]\n"
                f"Matriz de Confusión: [cyan]{cm_png}[/cyan]",
                border_style="magenta",
            )
        )
        console.print("[bold]Reporte de Clasificación por Subcarpeta (Test):[/bold]")
        console.print(cls_report_text)

        if test_eval["errors"]:
            err_tbl = Table(title=f"Archivos Confundidos en Test ({len(test_eval['errors'])})", box=box.SIMPLE)
            err_tbl.add_column("Archivo", style="yellow")
            err_tbl.add_column("Carpeta Real", style="green")
            err_tbl.add_column("Predicción", style="red")
            err_tbl.add_column("Confianza", justify="right")
            for err in test_eval["errors"][:15]:
                err_tbl.add_row(
                    err["file"],
                    node.class_names[err["true_idx"]],
                    node.class_names[err["pred_idx"]],
                    f"{err['confidence']:.1%}",
                )
            console.print(err_tbl)

        # Guardar primero SOLO en staging (nunca sobrescribe producción automáticamente)
        config.MODELS_STAGING_DIR.mkdir(parents=True, exist_ok=True)
        staging_model = config.MODELS_STAGING_DIR / f"{node.model_id}_candidate.keras"
        staging_map = config.MODELS_STAGING_DIR / f"{node.model_id}_class_map.json"

        model.save(staging_model)
        class_map_payload = {
            "model_id": node.model_id,
            "node_path": node.node_path,
            "level": node.level,
            "class_names": node.class_names,
            "image_size": list(config.IMAGE_SIZE),
            "resize_mode": config.RESIZE_MODE,
            "frames_per_file": args.frames,
            "backbone": config.BACKBONE_NAME,
            "alpha": args.alpha,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with staging_map.open("w", encoding="utf-8") as fh:
            json.dump(class_map_payload, fh, indent=2, ensure_ascii=False)

        with metrics_json.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_id": node.model_id,
                    "node_path": node.node_path,
                    "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "val_accuracy": round(val_eval["accuracy"], 4),
                    "val_macro_f1": round(val_eval["macro_f1"], 4),
                    "test_accuracy": round(test_eval["accuracy"], 4),
                    "test_macro_f1": round(test_eval["macro_f1"], 4),
                    "test_errors_count": len(test_eval["errors"]),
                    "classification_report": cls_report_text,
                    "history": combined_history,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        console.print(f"[green][OK][/green] Checkpoint candidato guardado en staging: [cyan]{staging_model}[/cyan]")

        # Puerta de aprobación explícita para promover a producción
        prod_model = config.MODELS_PROD_DIR / f"{node.model_id}.keras"
        prod_map = config.MODELS_PROD_DIR / f"{node.model_id}_class_map.json"
        if prod_model.exists():
            console.print(
                f"[bold yellow]AVISO:[/bold yellow] Ya existe un modelo en producción ({prod_model}). "
                "Solo se reemplazará si apruebas explícitamente."
            )

        if ask_yes_no(
            console,
            f"Tras revisar las métricas y la matriz de confusión, ¿apruebas promover '{node.model_id}' a producción?",
        ):
            config.MODELS_PROD_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging_model, prod_model)
            shutil.copy2(staging_map, prod_map)
            update_hierarchy_manifest(
                node=node,
                prod_model_path=prod_model,
                prod_class_map_path=prod_map,
                val_metrics=val_eval,
                test_metrics=test_eval,
            )
            console.print(
                f"[bold green][PROMOVIDO][/bold green] Modelo promovido a producción: [cyan]{prod_model}[/cyan]\n"
            )
        else:
            console.print(
                "[yellow]Promoción rechazada. El modelo candidato se conserva en staging "
                "y producción NO fue modificada.[/yellow]\n"
            )

        return True
    finally:
        # Limpieza estricta de RAM y caché temporal en disco
        _cleanup_cache_files(train_cache)
        _cleanup_cache_files(val_cache)
        if model is not None:
            del model
        try:
            import keras

            keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 3: Entrenamiento supervisado jerárquico con TensorBoard y Matriz de Confusión."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="ID del modelo jerárquico a entrenar (ej. imagenes, imagenes__capturas_pantalla, videos).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.DATASET_DIR,
        help=f"Carpeta raíz del dataset (por defecto: {config.DATASET_DIR}).",
    )
    parser.add_argument(
        "--max-cap",
        type=int,
        default=config.MAX_SAMPLES_PER_CLASS,
        help=f"Máximo de archivos por clase (por defecto: {config.MAX_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Tamaño de lote (por defecto: {config.BATCH_SIZE}, optimizado para 3.44 GB RAM).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS_PHASE1,
        help=f"Épocas máximas de Fase 1 (por defecto: {config.EPOCHS_PHASE1}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=config.VIDEO_FRAMES_PER_FILE,
        choices=[3, 4, 5],
        help=f"Fotogramas extraídos por video/animación (3 a 5; por defecto: {config.VIDEO_FRAMES_PER_FILE}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config.MOBILENET_ALPHA,
        choices=[0.35, 0.5, 0.75, 1.0],
        help=f"Multiplicador de ancho de MobileNetV2 (por defecto: {config.MOBILENET_ALPHA}).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Muestra únicamente los splits y pesos por clase sin iniciar entrenamiento.",
    )
    parser.add_argument(
        "--promote-staging",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help="Promueve interactivamente un modelo ya guardado en models/staging/ sin reentrenar.",
    )
    return parser.parse_args()


def promote_from_staging(model_id: str, console: Console) -> int:
    """Promueve un modelo candidato de staging a producción previa confirmación explícita."""
    staging_model = config.MODELS_STAGING_DIR / f"{model_id}_candidate.keras"
    staging_map = config.MODELS_STAGING_DIR / f"{model_id}_class_map.json"
    metrics_json = config.EVALUATIONS_DIR / f"{model_id}_metrics.json"

    if not staging_model.exists() or not staging_map.exists():
        console.print(
            f"[bold red]ERROR:[/bold red] No existe un candidato completo en staging para '{model_id}' "
            f"({staging_model})."
        )
        return 1

    with staging_map.open("r", encoding="utf-8") as fh:
        cmap = json.load(fh)

    val_acc, test_acc, test_f1 = 0.0, 0.0, 0.0
    if metrics_json.exists():
        with metrics_json.open("r", encoding="utf-8") as fh:
            mdata = json.load(fh)
            val_acc = float(mdata.get("val_accuracy", 0.0))
            test_acc = float(mdata.get("test_accuracy", 0.0))
            test_f1 = float(mdata.get("test_macro_f1", 0.0))

    console.print(
        Panel(
            f"[bold cyan]CANDIDATO EN STAGING:[/bold cyan] {model_id}\n"
            f"Nodo: {cmap['node_path']}/ (Nivel {cmap['level']}) | Clases: {', '.join(cmap['class_names'])}\n"
            f"Val Acc: [green]{val_acc:.2%}[/green] | Test Acc: [cyan]{test_acc:.2%}[/cyan] | "
            f"Test Macro-F1: [cyan]{test_f1:.2%}[/cyan]",
            border_style="cyan",
        )
    )

    if not sys.stdin.isatty():
        console.print("[bold red]BLOQUEADO:[/bold red] La promoción requiere confirmación interactiva.")
        return 1

    if not ask_yes_no(console, f"¿Confirmas promover '{model_id}' de staging a producción?"):
        console.print("[yellow]Promoción cancelada. Producción no fue modificada.[/yellow]")
        return 0

    config.MODELS_PROD_DIR.mkdir(parents=True, exist_ok=True)
    prod_model = config.MODELS_PROD_DIR / f"{model_id}.keras"
    prod_map = config.MODELS_PROD_DIR / f"{model_id}_class_map.json"
    shutil.copy2(staging_model, prod_model)
    shutil.copy2(staging_map, prod_map)

    dummy_node = NodeSpec(
        model_id=model_id,
        node_path=cmap["node_path"],
        level=int(cmap["level"]),
        class_names=list(cmap["class_names"]),
        files_by_class={},
    )
    update_hierarchy_manifest(
        node=dummy_node,
        prod_model_path=prod_model,
        prod_class_map_path=prod_map,
        val_metrics={"accuracy": val_acc},
        test_metrics={"accuracy": test_acc, "macro_f1": test_f1},
    )
    console.print(f"[bold green][PROMOVIDO][/bold green] Guardado en: [cyan]{prod_model}[/cyan]")
    return 0


def main() -> int:
    args = parse_args()
    console = Console()

    if args.promote_staging:
        return promote_from_staging(args.promote_staging, console)

    nodes = discover_trainable_nodes(args.dataset_dir.resolve())
    if not nodes:
        console.print("[bold red]ERROR:[/bold red] No se encontraron nodos jerárquicos entrenables.")
        return 1

    # Modo plan-only: muestra todos (o el elegido) y sale sin entrenar
    if args.plan_only:
        targets = [n for n in nodes if args.model is None or n.model_id == args.model]
        if not targets:
            console.print(f"[bold red]ERROR:[/bold red] Modelo no encontrado: {args.model}")
            return 1
        for node in targets:
            train_single_node(node, args, console)
        return 0

    # Si se especificó --model, ejecutar flujo supervisado para ese modelo
    if args.model:
        match = next((n for n in nodes if n.model_id == args.model), None)
        if match is None:
            valid_ids = ", ".join(n.model_id for n in nodes)
            console.print(f"[bold red]ERROR:[/bold red] Modelo '{args.model}' no existe. Opciones: {valid_ids}")
            return 1
        train_single_node(match, args, console)
        return 0

    # Menú interactivo para elegir qué nivel/modelo entrenar paso a paso
    if not sys.stdin.isatty():
        console.print(
            "[bold red]BLOQUEADO:[/bold red] Ejecuta este script en una terminal interactiva "
            "(o usa `--plan-only` para inspección)."
        )
        return 1

    while True:
        console.print()
        menu_tbl = Table(title="Clasificadores Jerárquicos Disponibles para Entrenamiento Supervisado", box=box.ROUNDED)
        menu_tbl.add_column("#", justify="right", style="bold cyan")
        menu_tbl.add_column("ID del Modelo", style="cyan")
        menu_tbl.add_column("Nivel", justify="center")
        menu_tbl.add_column("Clases", style="white")
        menu_tbl.add_column("Estado en Producción", style="green")

        for idx, n in enumerate(nodes, start=1):
            prod_exists = (config.MODELS_PROD_DIR / f"{n.model_id}.keras").exists()
            status = "PROMOVIDO (.keras)" if prod_exists else "Pendiente"
            menu_tbl.add_row(
                str(idx),
                n.model_id,
                str(n.level),
                ", ".join(f"{c}({len(n.files_by_class[c])})" for c in n.class_names),
                status,
            )

        console.print(menu_tbl)
        console.print(
            f"[dim]Rutas deterministas (sin modelo): {config.DETERMINISTIC_ROUTES}[/dim]\n"
            "Elige el número de un modelo para entrenar, o escribe [bold]0[/bold] para salir."
        )

        choice = console.input("[bold cyan]Selección: [/bold cyan]").strip()
        if choice in {"0", "q", "salir", "exit"}:
            console.print("[green]Saliendo del asistente de entrenamiento supervisado.[/green]")
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(nodes)):
            console.print("[red]Opción no válida.[/red]")
            continue

        selected = nodes[int(choice) - 1]
        train_single_node(selected, args, console)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
