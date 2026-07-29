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
