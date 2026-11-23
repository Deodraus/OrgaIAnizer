"""Organizador ligero (sin TensorFlow) para Documentos, Programas y Comprimidos/Descomprimidos.

Características:
- Documentos: organiza según las carpetas existentes del proyecto
  (`documentos/PDFs`, `documentos/WordS`, `documentos/Universidad`,
   `documentos/Universidad/Mallas Curriculares`, `documentos/Mi_Cursor`).
- Programas: mueve ejecutables e instaladores (`.exe`, `.msi`, `.msix`, `.appx`)
  a `descargas/programas/`.
- Comprimidos + Descompresión Inteligente (`.zip`, `.rar`, `.7z`, `.tar`, `.gz`, etc.):
  1. Guarda el comprimido en `descargas/comprimidos(zip,rar,etc)/`.
  2. Inspecciona su estructura interna (protegiendo contra Zip-Slip y contraseñas).
  3. Descomprime en `descargas/descomprimidos/`:
     - Si ya viene contenido en UNA sola carpeta raíz, conserva esa carpeta sin duplicar niveles.
     - Si tiene archivos sueltos o múltiples elementos en la raíz, crea automáticamente
       una carpeta con el nombre del comprimido y extrae todo adentro.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unicodedata
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src import config

NonMediaCategory = Literal["executable", "archive", "pdf", "word", "cursor", "unsupported"]

_JUNK_TOP_NAMES: frozenset[str] = frozenset({"__macosx", ".ds_store", "thumbs.db", "desktop.ini"})

# Palabras clave normalizadas (sin tildes, minúsculas) para subcarpetas de documentos
_MALLAS_FILENAME_KEYWORDS: tuple[str, ...] = (
    "malla",
    "plan_estudios",
    "plan de estudios",
    "plan-estudios",
    "pensum",
    "curricul",
)
_MALLAS_TEXT_KEYWORDS: tuple[str, ...] = (
    "malla curricular",
    "malla academica",
    "plan de estudios",
    "pensum",
)
_UNI_STRONG_KEYWORDS: tuple[str, ...] = (
    "universidades",
    "universidad",
    "pregrado",
    "semestre",
    "creditos",
)


@dataclass
class NonMediaDecision:
    """Decisión determinista/semántica para un archivo no multimedia."""

    category: NonMediaCategory
    target_rel_dir: str
    reason: str
    is_archive: bool = False


@dataclass
class ArchiveInspection:
    """Resultado de inspeccionar la estructura interna de un archivo comprimido."""

    ok: bool
    is_encrypted: bool = False
    is_unsafe_path: bool = False
    file_count: int = 0
    has_single_root_folder: bool = False
    single_root_name: str | None = None
    top_level_count: int = 0
    wrapper_folder_name: str = ""
    error_message: str = ""


@dataclass
class ExtractionResult:
    """Resultado de la descompresión inteligente en `descargas/descomprimidos/`."""

    ok: bool
    created_wrapper_folder: bool
    extracted_dest: Path | None
    message: str


def normalize_text(text: str) -> str:
    """Convierte a minúsculas, elimina tildes y normaliza separadores."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_like = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_like.lower()).strip()


def get_non_media_category(path: Path) -> NonMediaCategory:
    """Clasifica el tipo de archivo no multimedia por extensión."""
    ext = path.suffix.lower()
    if ext in config.EXECUTABLE_EXTENSIONS:
        return "executable"
    if ext in config.ARCHIVE_EXTENSIONS:
        return "archive"
    if ext in config.PDF_EXTENSIONS:
        return "pdf"
    if ext in config.WORD_EXTENSIONS:
        return "word"
    if ext in config.CURSOR_EXTENSIONS:
        return "cursor"
    return "unsupported"


def _archive_clean_stem(path: Path) -> str:
    """Obtiene un nombre limpio de carpeta para un archivo comprimido (maneja `.tar.gz`, etc.)."""
    name = path.name
    lower = name.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower.endswith(compound):
            stem = name[: -len(compound)]
            return stem.strip(" .") or "archivo_descomprimido"
    return path.stem.strip(" .") or "archivo_descomprimido"


def _extract_pdf_preview(path: Path, max_pages: int = 2, max_chars: int = 3500) -> str:
    """Lee texto ligero de las primeras páginas de un PDF sin cargar librerías pesadas."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return ""
        chunks: list[str] = []
        total = 0
        for page in reader.pages[:max_pages]:
            txt = page.extract_text() or ""
            chunks.append(txt)
            total += len(txt)
            if total >= max_chars:
                break
        return " ".join(chunks)[:max_chars]
    except Exception:
        return ""


def _extract_docx_preview(path: Path, max_chars: int = 3500) -> str:
    """Lee texto de un `.docx` directamente desde su XML interno usando `zipfile` nativo."""
    if path.suffix.lower() != ".docx":
        return ""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        texts: list[str] = []
        total = 0
        for elem in root.iter():
            if elem.tag.endswith("}t") and elem.text:
                texts.append(elem.text)
                total += len(elem.text)
                if total >= max_chars:
                    break
        return " ".join(texts)[:max_chars]
    except Exception:
        return ""


def classify_non_media_file(path: Path) -> NonMediaDecision | None:
    """Clasifica Documentos, Programas, Comprimidos y Cursores hacia su carpeta destino."""
    category = get_non_media_category(path)
    if category == "unsupported":
        return None

    if category == "executable":
        return NonMediaDecision(
            category="executable",
            target_rel_dir=config.DEST_PROGRAMAS_REL,
            reason=f"Ejecutable/instalador ({path.suffix.lower()})",
        )

    if category == "archive":
        return NonMediaDecision(
            category="archive",
            target_rel_dir=config.DEST_COMPRIMIDOS_REL,
            reason=f"Archivo comprimido ({path.suffix.lower()}) + descompresión automática",
            is_archive=True,
        )

    if category == "cursor":
        return NonMediaDecision(
            category="cursor",
            target_rel_dir=config.DEST_CURSORES_REL,
            reason=f"Archivo de cursor Windows ({path.suffix.lower()})",
        )

    # Documentos (.pdf / .docx / .doc / .odt / .rtf)
    fname_norm = normalize_text(path.stem)
    preview_raw = _extract_pdf_preview(path) if category == "pdf" else _extract_docx_preview(path)
    text_norm = normalize_text(preview_raw)

    # 1. Mallas curriculares / planes de estudio universitarios
    if any(kw in fname_norm for kw in _MALLAS_FILENAME_KEYWORDS):
        return NonMediaDecision(
            category=category,
            target_rel_dir=config.DEST_MALLAS_REL,
            reason="Documento de malla curricular / plan de estudios (por nombre)",
        )

    has_uni_inst = any(u in fname_norm for u in config.UNI_INSTITUTION_KEYWORDS)
    has_malla_text = any(kw in text_norm for kw in _MALLAS_TEXT_KEYWORDS)
    if has_malla_text or (has_uni_inst and ("ingenieria" in fname_norm or "sistemas" in fname_norm)):
        return NonMediaDecision(
            category=category,
            target_rel_dir=config.DEST_MALLAS_REL,
            reason="Documento de malla curricular / plan de estudios (por contenido)",
        )

    # 2. Documentos generales de Universidad (sin desviar liquidaciones/manuales a Universidad)
    is_receipt_or_manual = any(
        k in fname_norm for k in ("liquidacion", "certificado", "manual", "proyecto", "ensayo", "guia")
    )
    if not is_receipt_or_manual and (
        "universidad" in fname_norm
        or (sum(1 for k in _UNI_STRONG_KEYWORDS if k in text_norm) >= 3 and "creditos" in text_norm)
    ):
        return NonMediaDecision(
            category=category,
            target_rel_dir=config.DEST_UNIVERSIDAD_REL,
            reason="Documento universitario (por nombre/contenido)",
        )

    # 3. Enrutamiento estándar por formato de documento
    if category == "pdf":
        return NonMediaDecision(
            category="pdf",
            target_rel_dir=config.DEST_PDFS_REL,
            reason="Documento PDF estándar",
        )

    return NonMediaDecision(
        category="word",
        target_rel_dir=config.DEST_WORDS_REL,
        reason=f"Documento Word/texto ({path.suffix.lower()})",
    )


def _find_7z_executable() -> str | None:
    """Localiza `7z.exe` (7-Zip o NanaZip) incluso cuando se ejecuta sin PATH completo."""
    found = shutil.which("7z")
    if found:
        return found

    candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "7z.exe",
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _is_unsafe_archive_member(member_path: str) -> bool:
    """Detecta intentos de Zip-Slip (rutas absolutas o `..`)."""
    norm = member_path.replace("\\", "/").strip()
    if not norm:
        return False
    if norm.startswith("/") or re.match(r"^[a-zA-Z]:", norm):
        return True
    parts = [p for p in norm.split("/") if p]
    return any(p == ".." for p in parts)


def _analyze_entries(
    entries: list[tuple[str, bool, bool]],
    wrapper_name: str,
) -> ArchiveInspection:
    """Analiza la lista de entradas `(ruta, es_carpeta, esta_encriptado)` de un comprimido."""
    clean_entries: list[tuple[list[str], bool]] = []

    for raw_path, is_dir, is_enc in entries:
        if is_enc:
            return ArchiveInspection(
                ok=False,
                is_encrypted=True,
                wrapper_folder_name=wrapper_name,
                error_message="El archivo comprimido está protegido con contraseña.",
            )
        if _is_unsafe_archive_member(raw_path):
            return ArchiveInspection(
                ok=False,
                is_unsafe_path=True,
                wrapper_folder_name=wrapper_name,
                error_message=f"Ruta insegura detectada dentro del comprimido: {raw_path}",
            )

        norm = raw_path.replace("\\", "/").strip("/")
        if not norm:
            continue
        parts = [p for p in norm.split("/") if p and p != "."]
        if not parts:
            continue
        if parts[0].lower() in _JUNK_TOP_NAMES:
            continue
        clean_entries.append((parts, is_dir))

    if not clean_entries:
        return ArchiveInspection(
            ok=False,
            wrapper_folder_name=wrapper_name,
            error_message="El archivo comprimido está vacío.",
        )

    top_names_order: list[str] = []
    top_names_seen: set[str] = set()
    file_count = 0
    has_nested_under_top = False
    top_is_explicit_dir = False

    for parts, is_dir in clean_entries:
        if not is_dir:
            file_count += 1
        top = parts[0]
        if top not in top_names_seen:
            top_names_seen.add(top)
            top_names_order.append(top)
        if len(parts) > 1:
            has_nested_under_top = True
        elif is_dir:
            top_is_explicit_dir = True

    # Solo hay carpeta raíz única si TODOS los elementos cuelgan de 1 carpeta
    has_single_root = len(top_names_order) == 1 and (has_nested_under_top or top_is_explicit_dir)

    return ArchiveInspection(
        ok=True,
        file_count=file_count,
        has_single_root_folder=has_single_root,
        single_root_name=top_names_order[0] if has_single_root else None,
        top_level_count=len(top_names_order),
        wrapper_folder_name=wrapper_name,
    )


def _list_entries_zipfile(archive_path: Path) -> list[tuple[str, bool, bool]]:
    entries: list[tuple[str, bool, bool]] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            is_enc = bool(info.flag_bits & 0x1)
            entries.append((info.filename, info.is_dir(), is_enc))
    return entries


def _list_entries_7z(archive_path: Path, seven_z: str) -> list[tuple[str, bool, bool]]:
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(
        [seven_z, "l", "-slt", "-p", str(archive_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").lower()
        if "password" in err or "wrong password" in err or "encrypted" in err:
            return [("__encrypted__", False, True)]
        raise RuntimeError(f"7z falló al listar ({proc.returncode}): {(proc.stderr or proc.stdout).strip()[:200]}")

    entries: list[tuple[str, bool, bool]] = []
    in_listing = False
    cur_path: str | None = None
    cur_is_dir = False
    cur_is_enc = False

    def _flush() -> None:
        nonlocal cur_path, cur_is_dir, cur_is_enc
        if cur_path is not None:
            entries.append((cur_path, cur_is_dir, cur_is_enc))
        cur_path = None
        cur_is_dir = False
        cur_is_enc = False

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not in_listing:
            if line.startswith("----------"):
                in_listing = True
            continue

        if not line:
            _flush()
            continue

        if line.startswith("Path = "):
            cur_path = line[len("Path = ") :]
        elif line.startswith("Folder = "):
            cur_is_dir = line[len("Folder = ") :].strip() == "+"
        elif line.startswith("Attributes = ") and "D" in line[len("Attributes = ") :]:
            cur_is_dir = True
        elif line.startswith("Encrypted = "):
            cur_is_enc = line[len("Encrypted = ") :].strip() == "+"

    _flush()
    return entries


def inspect_archive(archive_path: Path) -> ArchiveInspection:
    """Inspecciona el contenido de `.zip`, `.rar`, `.7z`, `.tar`, etc. sin extraerlo."""
    wrapper_name = _archive_clean_stem(archive_path)
    ext = archive_path.suffix.lower()

    if ext == ".zip":
        try:
            entries = _list_entries_zipfile(archive_path)
            return _analyze_entries(entries, wrapper_name=wrapper_name)
        except Exception:
            pass  # Fallback a 7z si es un zip con compresión especial

    seven_z = _find_7z_executable()
    if seven_z is None:
        return ArchiveInspection(
            ok=False,
            wrapper_folder_name=wrapper_name,
            error_message=f"No se encontró 7z.exe para inspeccionar archivos {ext}.",
        )

    try:
        entries = _list_entries_7z(archive_path, seven_z)
        return _analyze_entries(entries, wrapper_name=wrapper_name)
    except Exception as exc:
        return ArchiveInspection(
            ok=False,
            wrapper_folder_name=wrapper_name,
            error_message=str(exc),
        )


def resolve_non_colliding_dir(parent_dir: Path, folder_name: str) -> Path:
    """Genera una ruta de carpeta destino sin sobrescribir ni mezclar carpetas existentes."""
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", folder_name).strip(" .") or "descomprimido"
    candidate = parent_dir / safe_name
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        alt = parent_dir / f"{safe_name}_{counter}"
        if not alt.exists():
            return alt
        counter += 1


def extract_archive_smart(
    archive_path: Path,
    descomprimidos_dir: Path,
) -> ExtractionResult:
    """Descomprime un archivo de forma inteligente y segura en `descargas/descomprimidos/`.

    - Si el comprimido ya tiene UNA sola carpeta raíz, conserva esa carpeta directamente.
    - Si contiene archivos sueltos o varias carpetas/archivos en la raíz, crea una carpeta
      nueva con el nombre del archivo comprimido y coloca todo adentro.
    """
    inspection = inspect_archive(archive_path)
    if not inspection.ok:
        return ExtractionResult(
            ok=False,
            created_wrapper_folder=False,
            extracted_dest=None,
            message=inspection.error_message,
        )

    descomprimidos_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = descomprimidos_dir / f".tmp_extract_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        ext = archive_path.suffix.lower()
        extracted_ok = False

        if ext == ".zip":
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(tmp_dir)
                extracted_ok = True
            except Exception:
                # Limpia intento parcial y usa 7z como respaldo
                shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)

        if not extracted_ok:
            seven_z = _find_7z_executable()
            if seven_z is None:
                return ExtractionResult(
                    ok=False,
                    created_wrapper_folder=False,
                    extracted_dest=None,
                    message=f"No se encontró 7z.exe para extraer {archive_path.name}.",
                )

            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.run(
                [seven_z, "x", "-y", "-p", f"-o{tmp_dir}", str(archive_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                check=False,
            )
            if proc.returncode != 0:
                err_tail = (proc.stderr or proc.stdout or "").strip()[-250:]
                return ExtractionResult(
                    ok=False,
                    created_wrapper_folder=False,
                    extracted_dest=None,
                    message=f"Error al extraer con 7z (código {proc.returncode}): {err_tail}",
                )

        # Eliminar basura __MACOSX si vino en el comprimido
        macosx = tmp_dir / "__MACOSX"
        if macosx.exists() and macosx.is_dir():
            shutil.rmtree(macosx, ignore_errors=True)

        real_children = [
            p for p in tmp_dir.iterdir() if p.name.lower() not in _JUNK_TOP_NAMES
        ]
        if not real_children:
            return ExtractionResult(
                ok=False,
                created_wrapper_folder=False,
                extracted_dest=None,
                message="La extracción no produjo archivos.",
            )

        # Caso 1: Ya venía en una sola carpeta raíz -> mover esa carpeta directo a descomprimidos/
        if len(real_children) == 1 and real_children[0].is_dir():
            root_folder = real_children[0]
            final_dest = resolve_non_colliding_dir(descomprimidos_dir, root_folder.name)
            shutil.move(str(root_folder), str(final_dest))
            return ExtractionResult(
                ok=True,
                created_wrapper_folder=False,
                extracted_dest=final_dest,
                message=f"Extraído conservando carpeta raíz '{final_dest.name}/' ({inspection.file_count} archivo(s)).",
            )

        # Caso 2: Archivos sueltos o múltiples elementos -> crear carpeta con el nombre del comprimido
        final_dest = resolve_non_colliding_dir(descomprimidos_dir, inspection.wrapper_folder_name)
        shutil.move(str(tmp_dir), str(final_dest))
        return ExtractionResult(
            ok=True,
            created_wrapper_folder=True,
            extracted_dest=final_dest,
            message=(
                f"Archivos sueltos detectados ({inspection.top_level_count} en raíz): "
                f"se creó la carpeta '{final_dest.name}/' ({inspection.file_count} archivo(s))."
            ),
        )
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
