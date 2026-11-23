"""Extractor multimedia unificado (imágenes estáticas, animadas y videos .mp4).

Características clave:
- Videos (.mp4): extrae entre 3 y 5 fotogramas distribuidos entre el 10% y el 85%
  de la duración mediante OpenCV (evitando pantallas negras iniciales/finales).
- Animaciones (.gif, .webp animado): extrae fotogramas clave distribuidos con Pillow.
- Imágenes estáticas (.jpg, .jpeg, .png, .webp, .gif): carga optimizada para poca RAM
  (usa `draft()` en JPEG para evitar picos de memoria en fotos de 108 MP).
- Salida estándar: tensores float32 de forma `(N, 224, 224, 3)` con el preprocesamiento
  exacto de `MobileNetV2` (escala `[-1.0, 1.0]`).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from src import config

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS_ALLOWED

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def get_media_type(path: Path) -> Literal["image", "video", "unsupported"]:
    """Clasifica determinísticamente el tipo de archivo por extensión (Nivel 1)."""
    ext = path.suffix.lower()
    if ext in config.IMAGE_EXTENSIONS:
        return "image"
    if ext in config.VIDEO_EXTENSIONS:
        return "video"
    return "unsupported"


def preprocess_mobilenet_v2(rgb_uint8: np.ndarray) -> np.ndarray:
    """Aplica el preprocesamiento exacto de `tf.keras.applications.mobilenet_v2.preprocess_input`.

    Convierte valores RGB `[0, 255]` a `float32` en `[-1.0, 1.0]`: `(x / 127.5) - 1.0`.
    """
    arr = rgb_uint8.astype(np.float32)
    arr /= 127.5
    arr -= 1.0
    return arr


def _compute_sample_indices(
    total_frames: int,
    num_frames: int,
    start_ratio: float = config.VIDEO_FRAME_START_RATIO,
    end_ratio: float = config.VIDEO_FRAME_END_RATIO,
) -> list[int]:
    """Calcula índices distribuidos uniformemente entre start_ratio y end_ratio."""
    if total_frames <= 1:
        return [0]

    num_frames = max(1, min(num_frames, total_frames))
    if total_frames <= num_frames:
        return list(range(total_frames))

    start_idx = int(total_frames * start_ratio)
    end_idx = int(total_frames * end_ratio) - 1
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_frames - 1

    raw_indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=int).tolist()
    # Deduplica manteniendo el orden
    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in raw_indices:
        clamped = max(0, min(int(idx), total_frames - 1))
        if clamped not in seen:
            seen.add(clamped)
            unique_indices.append(clamped)
    return unique_indices or [0]


def _pil_to_rgb_resized(img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Convierte una imagen PIL a RGB y redimensiona preservando proporción (letterbox)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        rgb = background
    elif img.mode != "RGB":
        rgb = img.convert("RGB")
    else:
        rgb = img

    target_w, target_h = target_size
    if config.RESIZE_MODE == "letterbox":
        orig_w, orig_h = rgb.size
        scale = min(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        if (new_w, new_h) != (orig_w, orig_h):
            rgb = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), config.LETTERBOX_PAD_COLOR)
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(rgb, offset)
        return np.asarray(canvas, dtype=np.uint8)

    if rgb.size != target_size:
        rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8)


def _cv2_rgb_letterbox(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Redimensiona un fotograma RGB de OpenCV preservando proporción (letterbox)."""
    target_w, target_h = target_size
    if config.RESIZE_MODE != "letterbox":
        return cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    orig_h, orig_w = rgb.shape[:2]
    scale = min(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), config.LETTERBOX_PAD_COLOR, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _extract_from_image(
    path: Path,
    num_frames: int,
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    """Extrae 1 fotograma (estática) o N fotogramas clave (GIF/WebP animado) como RGB uint8."""
    frames: list[np.ndarray] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as img:
            n_frames = getattr(img, "n_frames", 1)
            is_animated = bool(getattr(img, "is_animated", False) or n_frames > 1)

            if is_animated:
                indices = _compute_sample_indices(n_frames, num_frames)
                for idx in indices:
                    try:
                        img.seek(idx)
                        frame_copy = img.copy()
                        frames.append(_pil_to_rgb_resized(frame_copy, target_size))
                    except EOFError:
                        break
            else:
                # Optimización crítica de RAM para JPEGs enormes (ej. 108 MP):
                # libjpeg decodifica directamente a escala reducida sin expandir 324+ MB.
                if (img.format or "").upper() == "JPEG":
                    draft_box = (target_size[0] * 2, target_size[1] * 2)
                    try:
                        img.draft("RGB", draft_box)
                    except Exception:
                        pass

                img_oriented = ImageOps.exif_transpose(img)
                frames.append(_pil_to_rgb_resized(img_oriented, target_size))

    return frames


def _extract_from_video(
    path: Path,
    num_frames: int,
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    """Extrae N fotogramas entre el 10% y el 85% de la duración de un video .mp4."""
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV no pudo abrir el video: {path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            indices = _compute_sample_indices(total_frames, num_frames)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = _cv2_rgb_letterbox(rgb, target_size)
                frames.append(resized)

        # Fallback si el contenedor no reporta CAP_PROP_FRAME_COUNT o el seek falló
        if not frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = _cv2_rgb_letterbox(rgb, target_size)
                frames.append(resized)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No se pudo extraer ningún fotograma legible de: {path}")

    return frames


def extract_frames(
    path: str | Path,
    num_frames: int = config.VIDEO_FRAMES_PER_FILE,
    target_size: tuple[int, int] = config.IMAGE_SIZE,
    apply_preprocess: bool = True,
) -> np.ndarray:
    """Extrae y preprocesa fotogramas de cualquier archivo multimedia soportado.

    Args:
        path: Ruta al archivo (.jpg, .jpeg, .png, .webp, .gif, .mp4).
        num_frames: Cantidad de fotogramas a extraer en videos y animaciones (3 a 5).
        target_size: Tamaño de salida `(ancho, alto)`, por defecto `(224, 224)`.
        apply_preprocess: Si True, aplica el preprocesamiento exacto de MobileNetV2 `[-1, 1]`.

    Returns:
        Array NumPy de forma `(N, 224, 224, 3)` (`float32` si `apply_preprocess=True`,
        `uint8` si `apply_preprocess=False`).
    """
    file_path = Path(path)
    media_type = get_media_type(file_path)

    if media_type == "image":
        raw_frames = _extract_from_image(file_path, num_frames=num_frames, target_size=target_size)
    elif media_type == "video":
        raw_frames = _extract_from_video(file_path, num_frames=num_frames, target_size=target_size)
    else:
        raise ValueError(f"Formato no soportado: {file_path.suffix} ({file_path})")

    if not raw_frames:
        raise ValueError(f"No se extrajeron fotogramas de: {file_path}")

    stacked = np.stack(raw_frames, axis=0)
    if apply_preprocess:
        return preprocess_mobilenet_v2(stacked)
    return stacked
