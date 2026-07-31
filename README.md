# Gesture Matcher

Aplicación de escritorio en Python para reconocer señas estáticas a partir de la
geometría de landmarks de una o dos manos. El procesamiento será local y no
guardará fotogramas ni video automáticamente.

## Estado actual

El repositorio se encuentra en la Fase 2: cámara, en progreso. Actualmente incluye:

- paquete instalable con estructura `src`;
- configuración YAML tipada y validada;
- resolución segura de rutas relativas;
- logging centralizado;
- punto de entrada importable;
- servicio de cámara con resolución configurable, efecto espejo y medición de FPS;
- errores específicos y liberación segura de la captura;
- pruebas unitarias de configuración, recursos y cámara mediante mocks.

El servicio de cámara aún no está conectado al punto de entrada ni a una ventana de
video. MediaPipe, el reconocimiento y la visualización de landmarks todavía no están
implementados. El punto de entrada solo valida la base y reporta el modelo faltante.

## Requisitos

- Windows, Linux o macOS con cámara compatible con OpenCV para las fases futuras.
- Python 3.11 o 3.12.

El rango de Python se limita a las versiones que MediaPipe declara y prueba para
su distribución actual. No se considera compatible Python 3.14 en este proyecto.

## Instalación para desarrollo

En PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Las dependencias de ejecución están en `pyproject.toml`; `requirements.txt`
instala el proyecto en modo editable y `requirements-dev.txt` agrega Pytest y Ruff.

## Configuración

Los valores configurables viven en `config/config.yaml`. Las rutas de recursos
deben ser relativas a la raíz del proyecto. La carga rechaza campos ausentes,
tipos inválidos, probabilidades fuera del intervalo `[0, 1]`, resoluciones no
positivas y rutas que intenten salir del repositorio.

Las secciones disponibles son:

- `camera`: índice, resolución y modo espejo;
- `hand_detection`: máximo de manos y confianzas mínimas;
- `recognition`: similitud coseno, umbrales y tratamiento de lateralidad;
- `temporal_filter`: ventana y fotogramas necesarios para estabilidad;
- `display`: landmarks, FPS y tamaño de la imagen asociada;
- `resources`: modelo, plantillas, metadatos e imágenes;
- `logging`: nivel del registro técnico.

## Modelo de MediaPipe

El modelo no está incluido ni se descarga automáticamente. La ruta esperada es:

```text
models/hand_landmarker.task
```

Mientras falte, el punto de entrada mostrará una advertencia específica. La
validación estricta del modelo se incorporará junto con `HandDetector` en la Fase 3.

## Ejecución actual

Después de instalar el proyecto:

```powershell
python -m gesture_matcher.app
```

Por ahora este comando valida configuración y recursos básicos. No abre la cámara.

## Pruebas y estilo

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Las pruebas unitarias no requieren cámara física. `CameraService` recibe una fábrica
de captura inyectable para simular apertura, lectura, errores y liberación.

## Estructura

```text
assets/
config/config.yaml
data/
docs/
models/
scripts/
src/gesture_matcher/
|-- app.py
|-- camera/camera_service.py
|-- recognition/
|-- ui/
|-- utils/
`-- vision/
tests/
```

## Privacidad

El diseño mantiene el procesamiento local. La aplicación no debe guardar,
transmitir ni subir imágenes de la cámara sin una acción explícita del usuario.

## Próximo incremento

Conectar `CameraService` a un ciclo mínimo de video OpenCV, mostrar los FPS y
garantizar el cierre de la ventana y la cámara ante salida normal o errores. Después
se realizará la validación manual con una cámara física.
