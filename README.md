# Gesture Matcher

Aplicación de escritorio en Python para reconocer señas estáticas a partir de la
geometría de landmarks de una o dos manos. El procesamiento será local y no
guardará fotogramas ni video automáticamente.

## Estado actual

El repositorio se encuentra en la Fase 3: detección de manos, en progreso.
Actualmente incluye:

- paquete instalable con estructura `src`;
- configuración YAML tipada y validada;
- resolución segura de rutas relativas;
- logging centralizado;
- punto de entrada importable;
- servicio de cámara con resolución configurable, efecto espejo y medición de FPS;
- errores específicos y liberación segura de la captura;
- `HandDetector` basado en MediaPipe Tasks Hand Landmarker en modo video;
- detección configurable de una o dos manos, lateralidad y estado sin manos;
- dibujo de los 21 landmarks y sus conexiones;
- ventana OpenCV con FPS, cantidad de manos y salida mediante Q o ESC;
- pruebas unitarias de configuración, recursos, cámara, detección e interfaz.

La cámara, el detector y la ventana ya están conectados desde el punto de entrada.
El reconocimiento de señas todavía no está implementado y no se ejecuta ninguna
comparación o clasificación.

## Requisitos

- Windows, Linux o macOS con cámara compatible con OpenCV.
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

Mientras falte, `HandDetector` detendrá el inicio con un error específico que muestra
la ruta esperada. El modelo se carga una sola vez y se libera al cerrar la aplicación.

## Ejecución

Después de instalar el proyecto:

```powershell
python -m gesture_matcher.app
```

El comando carga el modelo, abre la cámara configurada y muestra el video con los
landmarks detectados. Presiona Q o ESC para cerrar. La cámara, MediaPipe y las
ventanas OpenCV se liberan tanto en la salida normal como ante errores.

Los parámetros `hand_detection.max_hands`, `display.show_landmarks` y
`display.show_fps` permiten seleccionar una o dos manos y activar o desactivar las
anotaciones correspondientes.

## Pruebas y estilo

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Las pruebas unitarias no requieren cámara física ni un modelo MediaPipe válido.
La cámara, Hand Landmarker, el reloj y las operaciones de ventana/dibujo tienen
adaptadores inyectables para simular detección, errores y liberación.

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
|-- ui/opencv_view.py
|-- utils/
`-- vision/
    |-- hand_detector.py
    `-- landmark_drawer.py
tests/
```

## Privacidad

El diseño mantiene el procesamiento local. La aplicación no debe guardar,
transmitir ni subir imágenes de la cámara sin una acción explícita del usuario.

## Próximo incremento

Agregar el modelo local y realizar la validación manual con cámara física para una y
dos manos, verificando resolución, espejo, lateralidad, FPS y cierre seguro. Solo
después de documentar esa validación corresponde iniciar la Fase 4 de normalización.
