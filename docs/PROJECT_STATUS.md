# Estado del proyecto

## Fase actual

Fase 6 — Motor de reconocimiento, completada.

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
- Ejecución real sobre 44 referencias: 42 aceptadas en cuatro clases y dos
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
- Auto-reconocimiento verificado de las 42 muestras persistidas en su clase
  correspondiente.

## Funcionalidades en progreso

- Ninguna; la Fase 7 todavía no se ha iniciado.

## Problemas conocidos

- `models/hand_landmarker.task` no está incluido.
- La normalización y extracción todavía no están conectadas al ciclo de video.
- Las imágenes `killua_pose_02.jpeg` y `kurapika_pose_05.jpeg` no produjeron
  detecciones y fueron excluidas de las plantillas.
- No hay imágenes de presentación asociadas en `assets/display_images/`.
- El reconocimiento todavía no está conectado al ciclo de video.
- La estabilización temporal todavía no está implementada.

## Pruebas pendientes

- Validación visual de cero y una mano, lateralidad y distintas orientaciones.
- Validación manual explícita de salida mediante Q y ESC.
- Prueba manual de error al desconectar la cámara durante el ciclo.
- Validación real del orden canónico con una clase de referencias de dos manos;
  actualmente las cuatro clases procesadas son de una mano.
- Validación manual del rechazo y de las puntuaciones usando vectores capturados en
  vivo, una vez que exista el filtro temporal.

## Próximo incremento recomendado

Iniciar la Fase 7 implementando `TemporalFilter` con ventana configurable,
confirmación por mayoría, reinicio del historial y tratamiento explícito del estado
desconocido.
