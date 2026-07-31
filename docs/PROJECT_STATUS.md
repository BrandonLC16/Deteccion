# Estado del proyecto

## Fase actual

Fase 3 — Detección de manos, en progreso.

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

## Funcionalidades en progreso

- Validación manual con el modelo Hand Landmarker y una cámara física.
- Comprobación visual de una y dos manos, lateralidad y landmarks.

## Problemas conocidos

- `models/hand_landmarker.task` no está incluido.
- La captura y la inferencia no se han validado con hardware y modelo reales en este
  entorno.
- La exactitud de lateralidad y landmarks aún requiere validación visual.
- No existen plantillas ni señas de ejemplo procesadas.
- La normalización y el reconocimiento todavía no están implementados.

## Pruebas pendientes

- Validación manual de apertura, resolución, espejo, FPS y cierre seguro.
- Validación visual de cero, una y dos manos con distintas orientaciones.
- Prueba manual de error al desconectar la cámara durante el ciclo.

## Próximo incremento recomendado

Agregar el modelo local y completar la validación manual de la Fase 3. Si la
detección y la liberación funcionan correctamente, iniciar la Fase 4 implementando
la normalización de landmarks con sus pruebas matemáticas.
