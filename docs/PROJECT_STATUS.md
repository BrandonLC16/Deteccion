# Estado del proyecto

## Fase actual

Fase 2 — Cámara, en progreso.

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

## Funcionalidades en progreso

- Integración de `CameraService` con el punto de entrada.
- Ciclo mínimo de video y cierre de la ventana de OpenCV.
- Validación manual de resolución, espejo y FPS con una cámara física.

## Problemas conocidos

- `models/hand_landmarker.task` no está incluido.
- El servicio de cámara aún no está conectado al ciclo de la aplicación.
- La captura no se ha validado con una cámara física en este entorno.
- El detector de manos aún no está implementado.
- No existen plantillas ni señas de ejemplo procesadas.

## Pruebas pendientes

- Validación manual con una cámara física de apertura, resolución, espejo y FPS.
- Pruebas del futuro ciclo de aplicación con la interfaz OpenCV simulada.

## Próximo incremento recomendado

Conectar `CameraService` a un ciclo mínimo de video OpenCV, mostrar los FPS y
garantizar la liberación de la cámara y las ventanas ante salida normal o errores.
