# OrgaIAnizer — Organizador Jerárquico Local de Archivos con IA

Sistema local de clasificación y organización automática de archivos para Windows (y Python multiplataforma) que combina:
* **Visión por Computadora Jerárquica (`TensorFlow / MobileNetV2`):** clasifica imágenes (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`) y videos (`.mp4`) con extracción de fotogramas, preprocesamiento *letterbox* (preserva relación de aspecto) y votación entre frames.
* **Entrenamiento 100% Supervisado y Visual:** inspección previa del balanceo de clases, monitoreo en vivo con **TensorBoard**, curvas de *Loss/Accuracy*, **Matriz de Confusión** y puerta de aprobación explícita antes de promover modelos de `staging/` a `production/`.
* **Organizador Rápido sin TensorFlow (< 1 s):** clasifica documentos (`.pdf`, `.docx`, `.doc`, `.odt`, `.rtf`), ejecutables (`.exe`, `.msi`, `.msix`, `.appx`), cursores (`.ani`, `.cur`) y **descomprime inteligentemente** archivos comprimidos (`.zip`, `.rar`, `.7z`, `.tar`, `.gz`, etc.) creando carpeta nueva solo cuando vienen archivos sueltos.
* **Vigilancia Multi-Carpeta en Segundo Plano (`watchdog`):** modo seguro `--dry-run` por defecto, umbral de confianza configurable (`80%`) y modo `--on-demand` optimizado para PCs de bajos recursos (CPU, desde **4 GB de RAM**; ~25 MB en reposo).

---

## Requisitos Previos

* **Python 3.10+** (con `pip`).
* **Opcional (para `.rar` y `.7z`):** [7-Zip](https://www.7-zip.org/) o [NanaZip](https://github.com/M2Team/NanaZip) instalado (`7z.exe`). Los archivos `.zip` funcionan de forma nativa sin dependencias externas.

### Instalación

```powershell
git clone https://github.com/Deodraus/OrgaIAnizer.git
cd OrgaIAnizer

# (Recomendado) Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```text
├── dataset/                     # Tus carpetas de entrenamiento (NO se sube a Git)
│   ├── imagenes/                # Ej. capturas_pantalla/, galeria/, imagenes_descargadas/
│   ├── videos/                  # Ej. galeria_videos/, Captures/, ...
│   ├── documentos/              # PDFs/, WordS/, Universidad/, Mi_Cursor/
│   └── descargas/               # programas/, comprimidos(zip,rar,etc)/, descomprimidos/
├── inbox/                       # Carpeta de entrada para pruebas (NO se sube a Git)
├── src/
│   ├── config.py                # Configuración general y carga de `user_rules.json`
│   ├── dataset_inspector.py     # Paso 1: escaneo, integridad y balanceo del dataset
│   ├── media_extractor.py       # Extractor unificado (OpenCV + Pillow + Letterbox 224x224)
│   ├── non_media_organizer.py   # Clasificador de documentos, programas y descompresor inteligente
│   ├── train_supervised.py      # Paso 2: entrenamiento jerárquico + TensorBoard + Matriz Confusión
│   └── organizer_watcher.py     # Paso 3: vigilante multi-carpeta + Dry-Run + modo --on-demand
├── scripts/
│   └── control_vigilante.ps1    # Control sin terminal en Windows (start/stop/status/startup)
├── user_rules.example.json      # Plantilla para personalizar rutas y palabras clave locales
└── requirements.txt
```

### Personalización local (opcional)
Si deseas personalizar rutas deterministas, carpetas destino o palabras clave universitarias sin modificar el código, copia `user_rules.example.json` a `user_rules.json` (ignorado por Git) y edítalo a tu gusto.

---

## Uso Paso a Paso

### 1. Inspeccionar tu dataset (solo lectura)
Crea tu estructura de carpetas dentro de `dataset/` y revisa el conteo, integridad y balanceo antes de entrenar:
```powershell
python -m src.dataset_inspector
```

### 2. Abrir TensorBoard en vivo (Terminal 1)
```powershell
python -m tensorboard.main --logdir logs/tensorboard
```
Abre `http://localhost:6006` en tu navegador.

### 3. Entrenar modelos jerárquicos de forma supervisada (Terminal 2)
El asistente te muestra los splits estratificados `70/15/15`, pide tu confirmación antes de entrenar, genera curvas y Matriz de Confusión, y solo promueve el modelo a `models/production/` si tú lo apruebas explícitamente:
```powershell
python -m src.train_supervised
```
*(Para PCs con CPU lenta, puedes añadir `--alpha 0.5` para usar una variante más liviana de MobileNetV2.)*

### 4. Probar el vigilante en Modo Seguro (`--dry-run` por defecto)
Copia archivos de prueba en `inbox/` y verifica a dónde movería cada archivo sin modificar nada:
```powershell
python -m src.organizer_watcher --once
```

### 5. Vigilancia multi-carpeta en segundo plano sin terminal (Windows)
Vigila simultáneamente `Pictures\Screenshots` (`Win + ImpPt`) y `Downloads` usando `pythonw.exe` y modo `--on-demand` (~25 MB de RAM en reposo). Funciona tanto en **CMD** como en **PowerShell**:

```cmd
:: Iniciar en Modo Seguro (Dry-Run)
powershell -ExecutionPolicy Bypass -File scripts\control_vigilante.ps1 start

:: Ver estado, uso de RAM y últimos registros del log
powershell -ExecutionPolicy Bypass -File scripts\control_vigilante.ps1 status

:: Detener el vigilante
powershell -ExecutionPolicy Bypass -File scripts\control_vigilante.ps1 stop

:: Iniciar en Modo Real (mueve archivos con confianza >= 80% y descomprime archivos)
powershell -ExecutionPolicy Bypass -File scripts\control_vigilante.ps1 start -Execute -Threshold 0.80

:: (Opcional) Iniciar automáticamente al encender Windows / desinstalarlo
powershell -ExecutionPolicy Bypass -File scripts\control_vigilante.ps1 install-startup -Execute
powershell -ExecutionPolicy Bypass -File scripts\control_vigilante.ps1 uninstall-startup
```

---

## Privacidad y Seguridad

* El archivo `.gitignore` excluye automáticamente `dataset/`, `inbox/`, `models/`, `logs/`, `reports/` y `user_rules.json` para que nunca se suban archivos personales, logs ni pesos entrenados al repositorio.
* El vigilante jamás sobrescribe archivos existentes (renombra automáticamente a `_1`, `_2`, ...) y protege la descompresión contra ataques *Zip-Slip* y archivos protegidos con contraseña.
