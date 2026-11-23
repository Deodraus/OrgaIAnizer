"""Paso 2: Escaneo, verificación de integridad y análisis de balanceo del dataset.

Este script opera en modo ESTRICTAMENTE SOLO LECTURA sobre la carpeta `dataset/`.
No modifica, mueve ni elimina ningún archivo, y no ejecuta ningún entrenamiento.

Uso:
    python -m src.dataset_inspector
    python -m src.dataset_inspector --no-check-integrity
    python -m src.dataset_inspector --max-cap 300 --no-prompt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageFile
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from src import config

# Evita falsos positivos con imágenes JPEG ligeramente truncadas solo en carga parcial;
# aquí mantenemos False para detectar archivos realmente dañados durante verify().
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Silencia logs ruidosos de OpenCV/FFmpeg en consola durante la inspección
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


@dataclass
class FolderStats:
    """Estadísticas de una carpeta dentro de las ramas multimedia."""

    rel_path: str
    name: str
    depth: int
    parent_rel: str | None
    children_rels: list[str] = field(default_factory=list)
    direct_valid_images: int = 0
    direct_valid_videos: int = 0
    direct_animated_images: int = 0
    direct_corrupt_files: list[dict[str, str]] = field(default_factory=list)
    direct_other_files: Counter[str] = field(default_factory=Counter)
    direct_system_ignored: int = 0
    direct_ext_counts: Counter[str] = field(default_factory=Counter)
    direct_size_bytes: int = 0

    # Totales acumulados (carpeta + todas sus subcarpetas)
    subtree_valid_images: int = 0
    subtree_valid_videos: int = 0
    subtree_corrupt_count: int = 0
    subtree_other_count: int = 0
    subtree_size_bytes: int = 0

    @property
    def direct_valid_total(self) -> int:
        return self.direct_valid_images + self.direct_valid_videos

    @property
    def subtree_valid_total(self) -> int:
        return self.subtree_valid_images + self.subtree_valid_videos


def verify_media_file(path: Path) -> tuple[bool, bool, str | None]:
    """Verifica la integridad básica de una imagen o video sin cargarlo completo en RAM.

    Returns:
        (es_valido, es_animado, mensaje_error)
    """
    ext = path.suffix.lower()
    try:
        if path.stat().st_size == 0:
            return False, False, "Archivo vacío (0 bytes)"
    except OSError as exc:
        return False, False, f"Error de lectura de sistema: {exc}"

    if ext in config.IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as img:
                n_frames = getattr(img, "n_frames", 1)
                is_animated = bool(getattr(img, "is_animated", False) or n_frames > 1)
                img.verify()
            return True, is_animated, None
        except Exception as exc:
            return False, False, f"Imagen corrupta o ilegible: {type(exc).__name__}: {exc}"

    if ext in config.VIDEO_EXTENSIONS:
        cap = None
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return False, False, "OpenCV no pudo abrir el video"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    return False, False, "Video sin fotogramas legibles"
            return True, True, None
        except Exception as exc:
            return False, False, f"Error al validar video: {type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()

    return False, False, f"Extensión no soportada: {ext}"


def _count_files_in_branch(branch_dir: Path) -> int:
    total = 0
    for _, _, filenames in os.walk(branch_dir):
        total += len(filenames)
    return total


def scan_multimedia_branch(
    dataset_root: Path,
    branch_name: str,
    check_integrity: bool,
    console: Console,
) -> dict[str, FolderStats]:
    """Escanea recursivamente una rama multimedia (solo lectura)."""
    branch_dir = dataset_root / branch_name
    stats_map: dict[str, FolderStats] = {}

    if not branch_dir.exists() or not branch_dir.is_dir():
        return stats_map

    total_files = _count_files_in_branch(branch_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"Escaneando {branch_name}/...", total=max(total_files, 1))

        for dirpath, dirnames, filenames in os.walk(branch_dir):
            dirnames.sort(key=lambda s: s.lower())
            filenames.sort(key=lambda s: s.lower())

            current_dir = Path(dirpath)
            rel_path = current_dir.relative_to(dataset_root).as_posix()
            parts = Path(rel_path).parts
            depth = len(parts)
            parent_rel = Path(*parts[:-1]).as_posix() if depth > 1 else None

            folder_stats = FolderStats(
                rel_path=rel_path,
                name=current_dir.name,
                depth=depth,
                parent_rel=parent_rel,
            )
            stats_map[rel_path] = folder_stats
            if parent_rel and parent_rel in stats_map:
                stats_map[parent_rel].children_rels.append(rel_path)

            for fname in filenames:
                fpath = current_dir / fname
                progress.advance(task_id, 1)

                if fname.lower() in config.IGNORED_SYSTEM_FILES:
                    folder_stats.direct_system_ignored += 1
                    continue

                ext = fpath.suffix.lower() or "<sin_ext>"
                try:
                    fsize = fpath.stat().st_size
                except OSError:
                    fsize = 0

                if ext in config.SUPPORTED_MEDIA_EXTENSIONS:
                    is_valid = True
                    is_animated = False
                    error_msg = None

                    if check_integrity:
                        is_valid, is_animated, error_msg = verify_media_file(fpath)
                    elif ext in config.VIDEO_EXTENSIONS:
                        is_animated = True

                    if is_valid:
                        folder_stats.direct_size_bytes += fsize
                        folder_stats.direct_ext_counts[ext] += 1
                        if ext in config.IMAGE_EXTENSIONS:
                            folder_stats.direct_valid_images += 1
                            if is_animated:
                                folder_stats.direct_animated_images += 1
                        else:
                            folder_stats.direct_valid_videos += 1
                    else:
                        folder_stats.direct_corrupt_files.append(
                            {
                                "path": fpath.relative_to(dataset_root).as_posix(),
                                "reason": error_msg or "Error desconocido",
                            }
                        )
                else:
                    folder_stats.direct_other_files[ext] += 1

    # Acumulación bottom-up (de hojas hacia la raíz)
    for rel in sorted(stats_map.keys(), key=lambda r: stats_map[r].depth, reverse=True):
        node = stats_map[rel]
        node.subtree_valid_images = node.direct_valid_images + sum(
            stats_map[c].subtree_valid_images for c in node.children_rels
        )
        node.subtree_valid_videos = node.direct_valid_videos + sum(
            stats_map[c].subtree_valid_videos for c in node.children_rels
        )
        node.subtree_corrupt_count = len(node.direct_corrupt_files) + sum(
            stats_map[c].subtree_corrupt_count for c in node.children_rels
        )
        node.subtree_other_count = sum(node.direct_other_files.values()) + sum(
            stats_map[c].subtree_other_count for c in node.children_rels
        )
        node.subtree_size_bytes = node.direct_size_bytes + sum(
            stats_map[c].subtree_size_bytes for c in node.children_rels
        )

    return stats_map


def scan_informative_branch(dataset_root: Path, branch_name: str) -> dict[str, Any]:
    """Escaneo rápido informativo de ramas no multimedia (documentos/descargas)."""
    branch_dir = dataset_root / branch_name
    if not branch_dir.exists() or not branch_dir.is_dir():
        return {"exists": False}

    subfolder_count = 0
    total_files = 0
    total_bytes = 0
    ext_counter: Counter[str] = Counter()
    level2_summary: dict[str, int] = {}

    for dirpath, dirnames, filenames in os.walk(branch_dir):
        current_dir = Path(dirpath)
        rel_parts = current_dir.relative_to(branch_dir).parts
        if current_dir != branch_dir:
            subfolder_count += 1

        top_sub = rel_parts[0] if rel_parts else "(raíz)"
        for fname in filenames:
            if fname.lower() in config.IGNORED_SYSTEM_FILES:
                continue
            fpath = current_dir / fname
            total_files += 1
            ext = fpath.suffix.lower() or "<sin_ext>"
            ext_counter[ext] += 1
            level2_summary[top_sub] = level2_summary.get(top_sub, 0) + 1
            try:
                total_bytes += fpath.stat().st_size
            except OSError:
                pass

    return {
        "exists": True,
        "branch": branch_name,
        "subfolder_count": subfolder_count,
        "total_files": total_files,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "top_extensions": ext_counter.most_common(10),
        "level2_counts": dict(sorted(level2_summary.items(), key=lambda kv: kv[1], reverse=True)),
    }


def analyze_hierarchical_models(
    all_media_stats: dict[str, FolderStats],
    suggested_cap: int,
) -> list[dict[str, Any]]:
    """Identifica cada nodo donde se entrenará un clasificador jerárquico y evalúa su balanceo."""
    model_nodes: list[dict[str, Any]] = []

    for rel_path in sorted(all_media_stats.keys(), key=lambda r: (all_media_stats[r].depth, r)):
        node = all_media_stats[rel_path]
        if not node.children_rels:
            continue

        classes_info: list[dict[str, Any]] = []
        empty_children: list[str] = []

        for child_rel in node.children_rels:
            child = all_media_stats[child_rel]
            total_valid = child.subtree_valid_total
            if total_valid == 0:
                empty_children.append(child.name)
            classes_info.append(
                {
                    "class_name": child.name,
                    "rel_path": child.rel_path,
                    "valid_total": total_valid,
                    "valid_images": child.subtree_valid_images,
                    "valid_videos": child.subtree_valid_videos,
                    "corrupt_count": child.subtree_corrupt_count,
                    "other_files_count": child.subtree_other_count,
                    "capped_total": min(total_valid, suggested_cap),
                }
            )

        non_empty_counts = [c["valid_total"] for c in classes_info if c["valid_total"] > 0]
        if len(non_empty_counts) < 2:
            # Nodo con menos de 2 clases pobladas: se registra igual con advertencia
            status = "NO_ENTRENABLE (<2 clases con datos)"
            ratio = None
            min_count = min(non_empty_counts) if non_empty_counts else 0
            max_count = max(non_empty_counts) if non_empty_counts else 0
        else:
            min_count = min(non_empty_counts)
            max_count = max(non_empty_counts)
            ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

            if min_count < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT:
                status = "CRITICO: MUY POCAS MUESTRAS (<7)"
            elif ratio > config.HIGH_IMBALANCE_RATIO:
                status = "DESBALANCEO ALTO"
            elif min_count < config.MIN_RECOMMENDED_SAMPLES:
                status = "POCAS MUESTRAS (<20)"
            elif ratio > config.MODERATE_IMBALANCE_RATIO:
                status = "DESBALANCEO MODERADO"
            else:
                status = "OK (BALANCEADO)"

        # Recomendaciones específicas por nodo
        recommendations: list[str] = []
        if node.direct_valid_total > 0:
            recommendations.append(
                f"Hay {node.direct_valid_total} archivo(s) sueltos directamente en '{rel_path}/' "
                f"(fuera de sus subcarpetas); decide si moverlos a una subcarpeta o ignorarlos."
            )
        if empty_children:
            recommendations.append(
                f"Subcarpeta(s) vacía(s) sin multimedia válida: {', '.join(empty_children)}."
            )
        if min_count < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT and len(non_empty_counts) >= 2:
            low_classes = [
                f"{c['class_name']} ({c['valid_total']})"
                for c in classes_info
                if 0 < c["valid_total"] < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT
            ]
            recommendations.append(
                f"Clase(s) con <{config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT} archivos: "
                f"{', '.join(low_classes)}. Un split estratificado 70/15/15 requiere al menos 7 muestras "
                f"(o agregar más ejemplos / fusionar clases)."
            )
        elif min_count < config.MIN_RECOMMENDED_SAMPLES and len(non_empty_counts) >= 2:
            low_classes = [
                f"{c['class_name']} ({c['valid_total']})"
                for c in classes_info
                if config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT
                <= c["valid_total"]
                < config.MIN_RECOMMENDED_SAMPLES
            ]
            recommendations.append(
                f"Clase(s) pequeñas (<{config.MIN_RECOMMENDED_SAMPLES}): {', '.join(low_classes)}. "
                f"Usa Data Augmentation + class_weight='balanced'."
            )

        if ratio is not None and ratio > config.MODERATE_IMBALANCE_RATIO:
            capped_counts = [c["capped_total"] for c in classes_info if c["capped_total"] > 0]
            capped_ratio = round(max(capped_counts) / min(capped_counts), 2) if capped_counts else ratio
            recommendations.append(
                f"Desbalanceo {ratio}x (con tope={suggested_cap} baja a {capped_ratio}x). "
                f"Activar class_weight='balanced' y/o ajustar tope máximo por clase."
            )

        # Detectar mezcla de tipos de medio dentro de una rama
        branch_root = rel_path.split("/")[0]
        if branch_root == "imagenes":
            vid_classes = [
                f"{c['class_name']} ({c['valid_videos']} videos)"
                for c in classes_info
                if c["valid_videos"] > 0
            ]
            if vid_classes:
                recommendations.append(
                    f"Contiene archivos .mp4 dentro de la rama imagenes/: {', '.join(vid_classes)}. "
                    f"Confirmar si deben entrenarse aquí o en la rama videos/."
                )
        elif branch_root == "videos":
            img_classes = [
                f"{c['class_name']} ({c['valid_images']} imágenes)"
                for c in classes_info
                if c["valid_images"] > 0
            ]
            if img_classes:
                recommendations.append(
                    f"Contiene imágenes dentro de la rama videos/: {', '.join(img_classes)}."
                )

        model_nodes.append(
            {
                "model_id": rel_path.replace("/", "__"),
                "node_path": rel_path,
                "level": node.depth + 1,  # Nivel que predice este modelo
                "num_classes_total": len(classes_info),
                "num_classes_populated": len(non_empty_counts),
                "total_valid_samples": sum(c["valid_total"] for c in classes_info),
                "loose_files_in_parent": node.direct_valid_total,
                "min_class_count": min_count,
                "max_class_count": max_count,
                "imbalance_ratio": ratio,
                "status": status,
                "classes": classes_info,
                "recommendations": recommendations,
            }
        )

    return model_nodes


def render_console_report(
    console: Console,
    all_media_stats: dict[str, FolderStats],
    model_nodes: list[dict[str, Any]],
    informative_summaries: list[dict[str, Any]],
    suggested_cap: int,
    check_integrity: bool,
) -> None:
    """Imprime un reporte visual completo en consola con `rich`."""
    console.print()
    console.print(
        Panel.fit(
            "[bold white]INSPECCIÓN PREVIA DEL DATASET (MODO SOLO LECTURA)[/bold white]\n"
            f"Ruta: [cyan]{config.DATASET_DIR}[/cyan]  |  "
            f"Verificación de integridad: [green]{'ACTIVA' if check_integrity else 'OMITIDA'}[/green]  |  "
            f"Tope simulado/clase: [yellow]{suggested_cap}[/yellow]",
            border_style="cyan",
        )
    )

    # 1. Árbol jerárquico multimedia
    tree = Tree("[bold cyan]dataset/ (Ramas Multimedia Visual)[/bold cyan]")
    tree_nodes: dict[str, Tree] = {}

    for rel in sorted(all_media_stats.keys(), key=lambda r: (all_media_stats[r].depth, r)):
        st = all_media_stats[rel]
        details = [f"[bold green]{st.subtree_valid_total}[/bold green] válidos"]
        if st.subtree_valid_images:
            details.append(f"{st.subtree_valid_images} img")
        if st.subtree_valid_videos:
            details.append(f"{st.subtree_valid_videos} vid")
        if st.direct_animated_images:
            details.append(f"{st.direct_animated_images} anim")
        if st.direct_valid_total > 0 and st.children_rels:
            details.append(f"[yellow]{st.direct_valid_total} sueltos aquí[/yellow]")
        if st.subtree_corrupt_count:
            details.append(f"[bold red]{st.subtree_corrupt_count} corruptos[/bold red]")
        if st.subtree_other_count:
            details.append(f"[dim]{st.subtree_other_count} otros[/dim]")

        label = f"[bold]{st.name}/[/bold]  ({', '.join(details)})"
        parent_tree = tree_nodes.get(st.parent_rel, tree) if st.parent_rel else tree
        tree_nodes[rel] = parent_tree.add(label)

    console.print(tree)
    console.print()

    # 2. Distribución por formato multimedia
    ext_totals: Counter[str] = Counter()
    other_totals: Counter[str] = Counter()
    corrupt_all: list[dict[str, str]] = []
    total_valid_img = 0
    total_valid_vid = 0
    total_anim_img = 0
    total_bytes = 0

    for st in all_media_stats.values():
        ext_totals.update(st.direct_ext_counts)
        other_totals.update(st.direct_other_files)
        corrupt_all.extend(st.direct_corrupt_files)
        total_valid_img += st.direct_valid_images
        total_valid_vid += st.direct_valid_videos
        total_anim_img += st.direct_animated_images
        total_bytes += st.direct_size_bytes

    fmt_table = Table(title="Resumen de Formatos Multimedia Detectados", box=box.ROUNDED)
    fmt_table.add_column("Extensión", style="cyan")
    fmt_table.add_column("Tipo", style="magenta")
    fmt_table.add_column("Cantidad Válida", justify="right", style="green")

    for ext, cnt in ext_totals.most_common():
        kind = "Imagen" if ext in config.IMAGE_EXTENSIONS else "Video"
        fmt_table.add_row(ext, kind, str(cnt))
    fmt_table.add_section()
    fmt_table.add_row(
        "[bold]TOTAL VÁLIDO[/bold]",
        f"{total_valid_img} img ({total_anim_img} anim) + {total_valid_vid} vid",
        f"[bold]{total_valid_img + total_valid_vid}[/bold]",
    )
    fmt_table.add_row("Peso multimedia total", "Solo archivos válidos", f"{total_bytes / (1024**2):.1f} MB")
    if corrupt_all:
        fmt_table.add_row("[bold red]Corruptos / Ilegibles[/bold red]", "Excluidos", f"[bold red]{len(corrupt_all)}[/bold red]")
    if other_totals:
        top_other = ", ".join(f"{e}({c})" for e, c in other_totals.most_common(6))
        fmt_table.add_row("[yellow]Otros en ramas media[/yellow]", top_other, f"[yellow]{sum(other_totals.values())}[/yellow]")

    console.print(fmt_table)
    console.print()

    # 3. Análisis por cada modelo jerárquico
    console.print("[bold underline]Análisis de Balanceo por Clasificador Jerárquico[/bold underline]\n")

    for m in model_nodes:
        status = m["status"]
        if status.startswith("OK"):
            status_style = "bold green"
        elif "CRITICO" in status or "NO_ENTRENABLE" in status:
            status_style = "bold red"
        elif "ALTO" in status:
            status_style = "bold red"
        else:
            status_style = "bold yellow"

        ratio_str = f"{m['imbalance_ratio']}x" if m["imbalance_ratio"] is not None else "N/A"
        tbl = Table(
            title=(
                f"Modelo [bold white]({m['model_id']})[/bold white] -> Nivel {m['level']} "
                f"(Total: {m['total_valid_samples']} | Ratio Max/Min: {ratio_str} | "
                f"[{status_style}]{status}[/{status_style}])"
            ),
            box=box.SIMPLE_HEAVY,
        )
        tbl.add_column("Clase (Subcarpeta)", style="cyan")
        tbl.add_column("Total", justify="right", style="bold")
        tbl.add_column("Img", justify="right")
        tbl.add_column("Vid", justify="right")
        tbl.add_column("% del Modelo", justify="right")
        tbl.add_column(f"Con Tope ({suggested_cap})", justify="right", style="yellow")
        tbl.add_column("Split 70/15/15 aprox.", justify="right", style="dim")

        total_m = max(m["total_valid_samples"], 1)
        for cls in sorted(m["classes"], key=lambda x: x["valid_total"], reverse=True):
            n = cls["valid_total"]
            capped = cls["capped_total"]
            pct = (n / total_m) * 100
            n_train = int(round(capped * config.TRAIN_RATIO))
            n_val = int(round(capped * config.VAL_RATIO))
            n_test = max(capped - n_train - n_val, 0)
            row_style = "red" if n < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT else ""
            tbl.add_row(
                cls["class_name"],
                str(n),
                str(cls["valid_images"]),
                str(cls["valid_videos"]),
                f"{pct:5.1f}%",
                str(capped),
                f"{n_train}/{n_val}/{n_test}",
                style=row_style,
            )
        console.print(tbl)

        if m["recommendations"]:
            for rec in m["recommendations"]:
                console.print(f"  [yellow][!][/yellow] {rec}")
        console.print()

    # 4. Detalle de corruptos (si los hubiera)
    if corrupt_all:
        c_tbl = Table(title=f"Archivos Corruptos o Ilegibles ({len(corrupt_all)})", box=box.ROUNDED)
        c_tbl.add_column("Archivo", style="red")
        c_tbl.add_column("Motivo")
        for item in corrupt_all[:25]:
            c_tbl.add_row(item["path"], item["reason"])
        if len(corrupt_all) > 25:
            c_tbl.add_row(f"... y {len(corrupt_all) - 25} más", "Ver reports/dataset_summary.json")
        console.print(c_tbl)
        console.print()

    # 5. Resumen informativo del resto del dataset (para el plan de mañana)
    if informative_summaries:
        info_tbl = Table(
            title="Resumen Informativo: Otras Ramas del Dataset (Reservadas para el Plan Integral de Mañana)",
            box=box.ROUNDED,
        )
        info_tbl.add_column("Rama", style="cyan")
        info_tbl.add_column("Subcarpetas", justify="right")
        info_tbl.add_column("Archivos", justify="right")
        info_tbl.add_column("Peso (MB)", justify="right")
        info_tbl.add_column("Top Extensiones")

        for item in informative_summaries:
            if not item.get("exists"):
                continue
            top_ext = ", ".join(f"{e}({c})" for e, c in item["top_extensions"][:6])
            info_tbl.add_row(
                f"{item['branch']}/",
                str(item["subfolder_count"]),
                str(item["total_files"]),
                f"{item['total_size_mb']:.1f}",
                top_ext or "-",
            )
        console.print(info_tbl)
        console.print()


def generate_distribution_plot(
    model_nodes: list[dict[str, Any]],
    suggested_cap: int,
    output_path: Path,
) -> bool:
    """Genera un gráfico PNG con la distribución de clases por modelo jerárquico."""
    if not model_nodes:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")  # Backend sin ventana (bajo consumo de RAM)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    n_models = len(model_nodes)
    fig_height = max(3.5 * n_models, 4.5)
    fig, axes = plt.subplots(n_models, 1, figsize=(11, fig_height), constrained_layout=True)
    if n_models == 1:
        axes = [axes]

    for ax, m in zip(axes, model_nodes):
        classes_sorted = sorted(m["classes"], key=lambda x: x["valid_total"])
        names = [c["class_name"] for c in classes_sorted]
        counts = [c["valid_total"] for c in classes_sorted]

        colors = []
        for cnt in counts:
            if cnt < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT:
                colors.append("#d9534f")  # Rojo: crítico
            elif cnt < config.MIN_RECOMMENDED_SAMPLES:
                colors.append("#f0ad4e")  # Amarillo: pocas muestras
            elif cnt > suggested_cap:
                colors.append("#5bc0de")  # Azul claro: se aplicará tope
            else:
                colors.append("#5cb85c")  # Verde: saludable

        bars = ax.barh(names, counts, color=colors, edgecolor="#333333", linewidth=0.6)
        ax.axvline(
            suggested_cap,
            color="#d9534f",
            linestyle="--",
            linewidth=1.2,
            label=f"Tope sugerido ({suggested_cap})",
        )
        ax.axvline(
            config.MIN_RECOMMENDED_SAMPLES,
            color="#f0ad4e",
            linestyle=":",
            linewidth=1.2,
            label=f"Mínimo recomendado ({config.MIN_RECOMMENDED_SAMPLES})",
        )

        max_val = max(counts) if counts else 1
        ax.set_xlim(0, max(max_val * 1.18, config.MIN_RECOMMENDED_SAMPLES + 5))
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_width() + max_val * 0.015,
                bar.get_y() + bar.get_height() / 2,
                str(cnt),
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ratio_str = f"{m['imbalance_ratio']}x" if m["imbalance_ratio"] is not None else "N/A"
        ax.set_title(
            f"Modelo: {m['node_path']}/ (Nivel {m['level']}) — {m['status']} | Ratio: {ratio_str}",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_xlabel("Cantidad de archivos válidos")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return True


def export_json_summary(
    all_media_stats: dict[str, FolderStats],
    model_nodes: list[dict[str, Any]],
    informative_summaries: list[dict[str, Any]],
    suggested_cap: int,
    check_integrity: bool,
    output_path: Path,
) -> None:
    """Guarda el reporte estructurado en JSON para revisión y uso en el Paso 3."""
    corrupt_files = []
    folders_serialized = {}
    for rel, st in sorted(all_media_stats.items()):
        corrupt_files.extend(st.direct_corrupt_files)
        folders_serialized[rel] = {
            "depth": st.depth,
            "parent": st.parent_rel,
            "children": st.children_rels,
            "direct_valid_images": st.direct_valid_images,
            "direct_valid_videos": st.direct_valid_videos,
            "direct_animated_images": st.direct_animated_images,
            "direct_corrupt_count": len(st.direct_corrupt_files),
            "direct_other_files": dict(st.direct_other_files),
            "direct_ext_counts": dict(st.direct_ext_counts),
            "subtree_valid_images": st.subtree_valid_images,
            "subtree_valid_videos": st.subtree_valid_videos,
            "subtree_valid_total": st.subtree_valid_total,
            "subtree_corrupt_count": st.subtree_corrupt_count,
            "subtree_size_mb": round(st.subtree_size_bytes / (1024 * 1024), 2),
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(config.DATASET_DIR),
        "integrity_checked": check_integrity,
        "suggested_cap_per_class": suggested_cap,
        "supported_extensions": {
            "images": sorted(config.IMAGE_EXTENSIONS),
            "videos": sorted(config.VIDEO_EXTENSIONS),
        },
        "hierarchical_models": model_nodes,
        "folders": folders_serialized,
        "corrupt_files": corrupt_files,
        "informative_branches": informative_summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 2: Inspección previa, verificación de integridad y balanceo del dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=config.DATASET_DIR,
        help=f"Carpeta raíz del dataset (por defecto: {config.DATASET_DIR})",
    )
    parser.add_argument(
        "--no-check-integrity",
        action="store_true",
        help="Omite la verificación de cabeceras con Pillow/OpenCV (escaneo más rápido).",
    )
    parser.add_argument(
        "--max-cap",
        type=int,
        default=config.MAX_SAMPLES_PER_CLASS,
        help=f"Tope máximo simulado de muestras por clase (por defecto: {config.MAX_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="No generar el gráfico PNG de distribución de clases.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Finaliza sin solicitar confirmación interactiva en consola.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()

    dataset_dir: Path = args.dataset_dir.resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        console.print(f"[bold red]ERROR:[/bold red] No existe la carpeta del dataset: {dataset_dir}")
        return 1

    check_integrity = not args.no_check_integrity

    # 1. Escanear ramas multimedia (imagenes/, videos/)
    all_media_stats: dict[str, FolderStats] = {}
    for branch in config.MULTIMEDIA_BRANCHES:
        branch_stats = scan_multimedia_branch(
            dataset_root=dataset_dir,
            branch_name=branch,
            check_integrity=check_integrity,
            console=console,
        )
        all_media_stats.update(branch_stats)

    if not all_media_stats:
        console.print(
            "[bold red]ERROR:[/bold red] No se encontraron las ramas multimedia esperadas "
            f"({', '.join(config.MULTIMEDIA_BRANCHES)}) dentro de {dataset_dir}."
        )
        return 1

    # 2. Analizar modelos jerárquicos y balanceo
    model_nodes = analyze_hierarchical_models(all_media_stats, suggested_cap=args.max_cap)

    # 3. Escaneo informativo rápido de otras ramas (documentos/, descargas/, etc.)
    informative_summaries: list[dict[str, Any]] = []
    other_top_dirs = sorted(
        d.name
        for d in dataset_dir.iterdir()
        if d.is_dir() and d.name not in config.MULTIMEDIA_BRANCHES
    )
    for branch in other_top_dirs:
        informative_summaries.append(scan_informative_branch(dataset_dir, branch))

    # 4. Mostrar reporte en consola
    render_console_report(
        console=console,
        all_media_stats=all_media_stats,
        model_nodes=model_nodes,
        informative_summaries=informative_summaries,
        suggested_cap=args.max_cap,
        check_integrity=check_integrity,
    )

    # 5. Exportar JSON y gráfico visual
    export_json_summary(
        all_media_stats=all_media_stats,
        model_nodes=model_nodes,
        informative_summaries=informative_summaries,
        suggested_cap=args.max_cap,
        check_integrity=check_integrity,
        output_path=config.DATASET_SUMMARY_JSON,
    )
    console.print(f"[green][OK][/green] Reporte JSON guardado en: [cyan]{config.DATASET_SUMMARY_JSON}[/cyan]")

    if not args.no_plot:
        plotted = generate_distribution_plot(
            model_nodes=model_nodes,
            suggested_cap=args.max_cap,
            output_path=config.DATASET_DISTRIBUTION_PNG,
        )
        if plotted:
            console.print(
                f"[green][OK][/green] Gráfico de distribución guardado en: "
                f"[cyan]{config.DATASET_DISTRIBUTION_PNG}[/cyan]"
            )
        else:
            console.print(
                "[yellow][!][/yellow] No se generó el gráfico PNG (instala `matplotlib` para habilitarlo)."
            )

    # 6. Punto de control humano (supervisión obligatoria antes del Paso 3)
    console.print()
    console.print(
        Panel(
            "[bold]Siguiente paso (Paso 3 - Entrenamiento Supervisado):[/bold]\n"
            "Revisa los conteos, el balanceo y las advertencias mostradas arriba.\n"
            "NO se ha iniciado ningún entrenamiento ni se ha modificado ningún archivo.",
            border_style="green",
        )
    )

    if not args.no_prompt and sys.stdin.isatty():
        answer = console.input(
            "\n¿Confirmas que revisaste este balanceo del dataset? (s/n): "
        ).strip().lower()
        if answer in {"s", "si", "sí", "y", "yes"}:
            console.print("[bold green]Confirmación registrada. Listo para avanzar al Paso 3.[/bold green]")
        else:
            console.print(
                "[yellow]Ajusta tus carpetas o parámetros (tope por clase / exclusiones) "
                "y vuelve a ejecutar `python -m src.dataset_inspector` antes del Paso 3.[/yellow]"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
