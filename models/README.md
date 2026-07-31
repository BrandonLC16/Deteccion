# Modelos locales

El detector MediaPipe Tasks Hand Landmarker requiere el modelo compatible en:

```text
models/hand_landmarker.task
```

El archivo binario no se incluye en este repositorio ni se descarga
automáticamente. Debe obtenerse desde la documentación oficial de MediaPipe y
copiarse con el nombre exacto `hand_landmarker.task`.

Si falta o MediaPipe no puede cargarlo, la aplicación termina antes de abrir la
cámara e informa la ruta involucrada. Los modelos `*.task` permanecen ignorados por
Git para evitar agregar binarios pesados accidentalmente.
