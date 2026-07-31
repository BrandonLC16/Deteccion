# Decisiones técnicas

## 2026-07-29 — Versiones de Python compatibles

### Problema

El repositorio contenía un entorno Python 3.11 versionado en la raíz y un entorno
ignorado creado con Python 3.14. MediaPipe no declara Python 3.14 entre las
versiones soportadas de su paquete.

### Alternativas consideradas

- Mantener Python 3.14 y aceptar una instalación no respaldada.
- Permitir cualquier Python 3 sin comprobar compatibilidad.
- Limitar el proyecto a Python 3.11 y 3.12.

### Decisión

Usar `requires-python = ">=3.11,<3.13"` y validar el desarrollo inicialmente con
Python 3.11.

### Motivo

Es el rango común entre el entorno disponible y las versiones publicadas y
probadas por MediaPipe, reduciendo fallos de instalación ajenos al código.

### Consecuencias

- Los entornos con Python 3.13 o superior deberán recrearse con una versión
  compatible.
- El rango podrá ampliarse cuando MediaPipe publique soporte verificable y la
  suite completa pase con esa versión.

## 2026-07-30 — Modo de ejecución de MediaPipe Hand Landmarker

### Problema

Hand Landmarker admite los modos `IMAGE`, `VIDEO` y `LIVE_STREAM`. El ciclo del MVP
necesita aprovechar el seguimiento entre fotogramas, conservar un resultado asociado
a cada imagen mostrada y mantener la implementación comprobable sin concurrencia.

### Alternativas consideradas

- Ejecutar cada fotograma como una imagen independiente con `detect`.
- Usar `detect_for_video` de forma síncrona en modo `VIDEO`.
- Usar `detect_async` en modo `LIVE_STREAM` con callbacks y descarte de fotogramas.

### Decisión

Usar Hand Landmarker en modo `VIDEO` y llamar `detect_for_video` con marcas de tiempo
monotónicas y estrictamente crecientes generadas por `HandDetector`.

### Motivo

El modo `VIDEO` habilita el seguimiento de MediaPipe y entrega un resultado por cada
fotograma procesado. Evita introducir callbacks, estado compartido o descarte de
fotogramas antes de medir que sean necesarios para el MVP.

### Consecuencias

- La inferencia se ejecuta de forma síncrona dentro del ciclo de video.
- Si el rendimiento medido bloquea la interfaz, podrá reevaluarse `LIVE_STREAM` o una
  separación explícita de captura, inferencia e interfaz.
- El detector valida timestamps para impedir llamadas no crecientes rechazadas por
  la API de MediaPipe.
