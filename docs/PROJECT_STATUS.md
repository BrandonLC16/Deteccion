# Estado del proyecto

## Fase actual

Fase 1 — Base del proyecto, completada.

## Funcionalidades terminadas

- Estructura modular inicial bajo `src/gesture_matcher`.
- Empaquetado editable y dependencias declaradas.
- Configuración YAML con dataclasses inmutables y validación estricta.
- Resolución segura de rutas relativas al proyecto.
- Configuración central de logging.
- Punto de entrada que informa con claridad si falta el modelo.
- Directorios separados para referencias, presentación, datos y modelos.

## Funcionalidades en progreso

- Ninguna dentro de este incremento.

## Problemas conocidos

- `models/hand_landmarker.task` no está incluido.
- La cámara y el detector de manos aún no están implementados.
- No existen plantillas ni señas de ejemplo procesadas.

## Pruebas pendientes

- Las pruebas de cámara con mocks se agregarán en la Fase 2.
- La validación manual con cámara física comenzará después de implementar su servicio.

## Próximo incremento recomendado

Implementar y probar `CameraService`, asegurando apertura configurable, efecto
espejo y liberación de recursos incluso ante errores.
