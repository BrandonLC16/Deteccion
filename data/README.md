# Datos generados

El comando `python scripts/build_templates.py` genera aquí:

- `gesture_templates.npz`
- `gestures.json`

El archivo NPZ guarda una matriz `float32` por identificador de seña. Cada fila
es una muestra: 63 valores para una mano o 129 para dos manos en el orden
canónico izquierda-derecha.

El archivo JSON conserva el formato versionado, las etiquetas, la cantidad de
manos, la lateralidad, las rutas relativas y el umbral opcional de cada seña.
`TemplateRepository` valida ambos archivos y exige que sus claves, cantidades y
dimensiones coincidan antes de exponer las plantillas al motor.

Ambos archivos son artefactos generados, están excluidos de Git y deben
regenerarse desde las imágenes de referencia en vez de editarse manualmente.
