# Imágenes de presentación

Este directorio contiene únicamente las imágenes mostradas cuando se confirma una
seña. No se deben mezclar con las muestras de `reference_images`.

El nombre base debe coincidir con el `gesture_id`, por ejemplo `gon_pose.jpg`. Se
admiten las mismas extensiones de imagen del constructor de plantillas. Después de
agregar o reemplazar una imagen, ejecuta `python scripts/build_templates.py` para
actualizar `display_image_path` en los metadatos.

La aplicación carga la imagen desde esa ruta una sola vez, la conserva en caché y
la ajusta al panel sin deformar su proporción.
