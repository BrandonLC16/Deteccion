# Scripts del proyecto

## Construir plantillas

Ejecutar desde la raíz, con el proyecto y sus dependencias instalados:

```powershell
python scripts/build_templates.py
```

El script recorre cada carpeta de `assets/reference_images/`, valida extensiones y
decodificación, detecta manos con MediaPipe en modo `IMAGE` y genera:

- `data/gesture_templates.npz`: una matriz `float32` por identificador de seña;
- `data/gestures.json`: formato, cantidad de manos, lateralidades, rutas de origen,
  umbrales y relación entre cada fila de la matriz y su imagen.

Cada carpeta de seña debe usar `snake_case`. En señas de dos manos se exige una
mano `Left` y una `Right`; el orden persistido siempre es Left seguido de Right.
Al terminar se muestran todas las imágenes aceptadas y rechazadas con el motivo
correspondiente. El proceso no modifica los archivos originales.
