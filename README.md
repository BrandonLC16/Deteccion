# Gesture Matcher

Aplicación de escritorio en Python para reconocer señas estáticas a partir de la
geometría de landmarks de una o dos manos. El procesamiento será local y no
guardará fotogramas ni video automáticamente.

## Estado actual

El repositorio implementó la Fase 8: visualización del resultado.
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
- normalización por muñeca y escala para los 21 landmarks de cada mano;
- canonicalización opcional de manos izquierdas mediante reflexión del eje X;
- extracción de vectores geométricos inmutables de 63 valores;
- pruebas de invariancia a traslación, escala y lateralidad, además de entradas
  inválidas.
- detección offline de manos con MediaPipe en modo imagen;
- recorrido y validación de carpetas de referencias;
- orden canónico Left seguido de Right para señas de dos manos;
- persistencia comprimida de plantillas NPZ y metadatos JSON versionados;
- reporte detallado de imágenes aceptadas y rechazadas.
- carga validada de plantillas y recursos asociados;
- ranking por similitud coseno contra todas las muestras compatibles;
- umbral global y umbrales individuales por seña;
- resultado desconocido seguro para entradas vacías, inválidas o insuficientes;
- filtrado estricto por cantidad de manos y dimensión.
- ventana temporal configurable con confirmación por dominancia y racha mínima;
- retención breve del último resultado e histéresis para evitar alternancias;
- reinicio explícito del historial y conservación estable de la imagen asociada.
- pipeline conectado de detección, extracción, comparación y estabilización;
- panel principal compacto y centrado con reparto horizontal 38/62 para cámara e
  imagen;
- caché de imágenes con ajuste proporcional y reutilización entre fotogramas;
- estado visible `Seña desconocida` mientras no exista una coincidencia confirmada.

La cámara, el detector y la ventana ya están conectados desde el punto de entrada.
La extracción, el motor de reconocimiento, el filtro temporal y el panel visual se
ejecutan dentro del ciclo de video. Las imágenes de referencia se procesan únicamente
mediante un script explícito; las plantillas y cada imagen de presentación se cargan
una sola vez y se reutilizan en memoria.

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
- `temporal_filter`: ventana, votos, racha consecutiva, retención e histéresis;
- `display`: landmarks, FPS y tamaño de la imagen asociada;
- `resources`: modelo, plantillas, metadatos e imágenes;
- `logging`: nivel del registro técnico.

## Construcción de plantillas

Cada carpeta hija de `assets/reference_images/` representa una seña y debe usar un
identificador `snake_case`. Se admiten archivos BMP, JPEG, JPG, PNG y WebP. Ejecuta
`python scripts/build_templates.py` para construir las plantillas.

El script carga el modelo una sola vez en modo `IMAGE`, valida cada archivo, detecta
una o dos manos, extrae sus características y escribe
`data/gesture_templates.npz` y `data/gestures.json`.

Para dos manos, el orden siempre es `Left` seguido de `Right`, sin depender del
orden devuelto por MediaPipe. El vector también agrega el desplazamiento relativo
entre muñecas normalizado por el tamaño medio de las manos. Las imágenes sin manos,
ilegibles, con extensión no permitida o con lateralidad ambigua se rechazan con un
motivo específico. Las imágenes originales nunca se modifican.

## Motor de reconocimiento

`TemplateRepository` carga y valida conjuntamente el NPZ y el JSON. Rechaza
versiones incompatibles, matrices corruptas, dimensiones incorrectas, valores no
finitos y recursos asociados inexistentes.

`GestureMatcher.match(features)` recibe un vector de 63 valores para una mano o
129 para dos. Calcula similitud coseno contra todas las muestras con la misma
cantidad de manos, conserva la mejor puntuación por seña y ordena el ranking de
mayor a menor. Una coincidencia se acepta solo cuando alcanza su umbral individual
o, en su ausencia, el umbral global.

Las puntuaciones de cada muestra se registran con nivel `DEBUG`. Una entrada vacía,
no finita, de norma cero o con dimensión incompatible devuelve `Seña desconocida`
con puntuación `0.0`, sin lanzar excepciones.

## Estabilización temporal

`TemporalFilter.update(result)` conserva una ventana de resultados del motor. Una
seña se confirma cuando tiene una mayoría única, alcanza `stable_frames` votos y
también aparece durante `min_consecutive_frames` al final de la ventana.

Después de confirmar, `hold_frames` conserva brevemente el último resultado ante
ausencias. `hysteresis_frames` reduce el umbral necesario para mantener una seña ya
activa, pero no el requerido para activar una nueva. Una seña diferente solo toma
el control cuando se vuelve dominante y cumple su propia racha consecutiva. Esto
mantiene estable también `display_image_path` frente a variaciones aisladas.

`TemporalFilter.reset()` elimina la ventana y obliga a confirmar nuevamente. El
filtro trabaja únicamente con resultados aceptados por `GestureMatcher`; por ello,
el control previo de una o dos manos permanece vigente.

## Visualización del resultado

`RecognitionPipeline` conecta landmarks, extracción, comparación y estabilización.
`OpenCVView` muestra un panel compacto centrado dentro de una ventana redimensionable.
La franja superior asigna el 38 % del ancho útil a la cámara y el 62 % a la imagen
asociada; ambas secciones conservan la misma altura y la etiqueta y el porcentaje
de similitud quedan debajo de ellas.
El panel parte de una ventana de 1024 × 640 píxeles y limita su contenido a 920 × 560
píxeles, por lo que conserva márgenes visibles y no crece hasta ocupar toda la
pantalla.

La geometría se recalcula cuando cambia el tamaño útil de la ventana. Tanto el video
ya procesado como la imagen de presentación se ajustan dentro de su sección sin
recortar ni deformar su relación de aspecto. Este ajuste ocurre después de la
detección y del dibujo de landmarks: no modifica la resolución de cámara ni la
entrada utilizada por MediaPipe.

La imagen solo se solicita a `ImageCache` cuando el resultado estabilizado tiene
`accepted=True`. La caché lee cada ruta de `assets/display_images/` una vez —también
recuerda fallos— y ajusta la imagen dentro del tamaño configurado sin alterar su
proporción. No utiliza imágenes de `assets/reference_images/` como presentación.

Cuando el resultado todavía no está confirmado, el panel muestra `Seña desconocida`
y ninguna imagen. Si una plantilla aceptada no tiene imagen disponible, mantiene la
etiqueta y puntuación e informa `Sin imagen asociada`.

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

El comando carga modelo, plantillas y caché, abre la cámara configurada y muestra el
video con landmarks y el panel de reconocimiento. Presiona Q o ESC para cerrar. La
cámara, MediaPipe y las ventanas OpenCV se liberan tanto en la salida normal como
ante errores.

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
La cámara, Hand Landmarker, el reloj, el detector offline y las operaciones de
ventana/dibujo tienen adaptadores inyectables para simular detección, errores y
liberación.

## Estructura

```text
assets/
config/config.yaml
data/
docs/
models/
scripts/build_templates.py
src/gesture_matcher/
|-- app.py
|-- camera/camera_service.py
|-- recognition/
|   |-- feature_extractor.py
|   |-- gesture_matcher.py
|   |-- landmark_normalizer.py
|   |-- recognition_pipeline.py
|   |-- template_builder.py
|   |-- temporal_filter.py
|   `-- template_repository.py
|-- ui/
|   |-- image_overlay.py
|   `-- opencv_view.py
|-- utils/
`-- vision/
    |-- hand_detector.py
    |-- image_hand_detector.py
    `-- landmark_drawer.py
tests/
```

## Privacidad

El diseño mantiene el procesamiento local. La aplicación no debe guardar,
transmitir ni subir imágenes de la cámara sin una acción explícita del usuario.

## Próximo incremento

Validar visualmente el panel y calibrar umbrales con la cámara real. Después, la
Fase 9 puede evaluar una interfaz avanzada únicamente si el MVP mantiene estabilidad
y rendimiento suficientes.
