# Estado del proyecto

## Fase actual

Fase 8 — Visualización del resultado, implementada con validación manual pendiente.

## Funcionalidades terminadas

- Estructura modular inicial bajo `src/gesture_matcher`.
- Empaquetado editable y dependencias declaradas.
- Configuración YAML con dataclasses inmutables y validación estricta.
- Resolución segura de rutas relativas al proyecto.
- Configuración central de logging.
- Punto de entrada que informa con claridad si falta el modelo.
- Directorios separados para referencias, presentación, datos y modelos.
- `CameraService` con apertura por índice y resolución configurables.
- Lectura con efecto espejo opcional y medición de FPS.
- Errores específicos de apertura y lectura.
- Liberación idempotente y administrador de contexto para cierre seguro.
- Pruebas de cámara mediante mocks, sin depender de hardware físico.
- Integración de `CameraService` con el ciclo de aplicación y la ventana OpenCV.
- `HandDetector` con MediaPipe Tasks Hand Landmarker en modo `VIDEO`.
- Marcas de tiempo estrictamente crecientes para cada fotograma.
- Detección configurable de una o dos manos, lateralidad y estado sin manos.
- Conversión del resultado de MediaPipe a dataclasses propias del dominio.
- Dibujo de los 21 landmarks, conexiones y lateralidad.
- Visualización de FPS y cantidad de manos; salida mediante Q o ESC.
- Liberación garantizada de cámara, detector y ventanas mediante contextos.
- Pruebas del detector, dibujador, vista y ciclo integrado mediante mocks.
- Validación física reportada de detección simultánea de dos manos sin cierre
  inesperado de la aplicación.
- `LandmarkNormalizer` con muñeca como origen, escala euclidiana estable y
  canonicalización configurable de lateralidad.
- `FeatureExtractor` con vectores por mano de 63 coordenadas normalizadas.
- Validaciones de forma, valores finitos y escala no degenerada.
- Pruebas matemáticas de invariancia a traslación, escala y lateralidad.
- Detector offline de MediaPipe en modo `IMAGE`, independiente del detector de
  video.
- Orden canónico Left seguido de Right para señas de dos manos.
- Características relativas de muñecas para conservar la disposición entre dos
  manos.
- Recorrido de carpetas, validación de imágenes y rechazo con motivos específicos.
- Persistencia atómica por archivo en NPZ comprimido y JSON versionado.
- Reporte de clases e imágenes aceptadas y rechazadas.
- Ejecución real sobre 55 referencias: 53 aceptadas en cinco clases y dos
  rechazadas por ausencia de manos detectables.
- `TemplateRepository` con carga segura de NPZ/JSON, validación de versión,
  dimensiones, muestras, lateralidades, umbrales y recursos asociados.
- `GestureMatcher` con similitud coseno contra todas las muestras compatibles.
- Ranking determinista, mejor muestra por seña y rechazo por umbral.
- Umbral global, umbral persistido por seña y sobrescritura configurable en tiempo
  de ejecución.
- Resultado desconocido con puntuación para entradas vacías, inválidas, de norma
  cero o con cantidad de manos incompatible.
- Registro de puntuaciones por muestra en nivel `DEBUG`.
- Auto-reconocimiento verificado de las 53 muestras persistidas en su clase
  correspondiente.
- Clase real `gon_pose` validada con 11 muestras de dos manos, orden canónico
  Left-Right y vectores de 129 valores.
- `TemporalFilter` con ventana acotada, mayoría única y racha consecutiva mínima.
- Retención breve de la última seña confirmada e histéresis para absorber
  variaciones sin alternar la imagen asociada.
- Cambio de seña únicamente cuando la candidata se vuelve dominante y estable.
- Reinicio explícito del historial y tratamiento seguro del estado desconocido.
- `RecognitionPipeline` conectado al ciclo de video para cero, una o dos manos.
- Panel OpenCV compacto, centrado y responsivo con cuadros del mismo tamaño para
  cámara e imagen, etiqueta y porcentaje de similitud.
- Estado `Seña desconocida` visible hasta que el filtro confirme una coincidencia.
- `ImageCache` con lectura única por ruta, caché de fallos y rutas limitadas al
  proyecto.
- Redimensionado proporcional con márgenes para evitar deformar imágenes.
- Adaptación al tamaño útil de la ventana sin modificar la resolución de cámara ni
  la entrada del detector.
- Cinco imágenes de presentación enlazadas en los metadatos reconstruidos.
- Aparición de `gon_pose.jpg` validada a partir del quinto resultado confirmado y
  retención ante un fotograma transitorio de una mano.

## Funcionalidades en progreso

- Validación visual manual del panel con la cámara física.

## Problemas conocidos

- `models/hand_landmarker.task` no está incluido.
- Las imágenes `killua_pose_02.jpeg` y `kurapika_pose_05.jpeg` no produjeron
  detecciones y fueron excluidas de las plantillas.

## Pruebas pendientes

- Validación visual de cero y una mano, lateralidad y distintas orientaciones.
- Validación manual explícita de salida mediante Q y ESC.
- Prueba manual de error al desconectar la cámara durante el ciclo.
- Validación manual del rechazo y de las puntuaciones usando vectores capturados en
  vivo.
- Validación visual de la estabilidad temporal y permanencia de la imagen asociada
  dentro del ciclo de cámara.
- Medición de FPS con el panel y la caché activos.

## Próximo incremento recomendado

Ejecutar la validación visual y calibración del MVP con cámara real. Si etiqueta,
imagen, rechazo, estabilidad y rendimiento son adecuados, evaluar la Fase 9 de
interfaz avanzada sin acoplar la lógica de reconocimiento a la UI.
