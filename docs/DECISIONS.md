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

## 2026-07-31 — Normalización y vector de características por mano

### Problema

Los landmarks dependen de la posición y el tamaño aparente de la mano. Además, una
misma geometría aparece reflejada entre manos izquierdas y derechas, por lo que no
puede compararse de forma consistente sin definir un origen, una escala y una regla
de lateralidad.

### Alternativas consideradas

- Centrar en la muñeca y escalar por la distancia máxima desde ella.
- Escalar con una única distancia anatómica, como muñeca a dedo medio.
- Conservar la lateralidad o reflejar las manos izquierdas a una geometría canónica.
- Usar directamente las coordenadas normalizadas o aplanarlas en orden MediaPipe.

### Decisión

Para cada mano, restar la muñeca a los 21 landmarks y dividir las coordenadas por la
mayor distancia euclidiana desde la muñeca. Cuando `mirror_left_hand` esté activo,
reflejar el eje X de las manos `Left`. Aplanar después las coordenadas X, Y, Z en el
orden de landmarks de MediaPipe para producir un vector `float32` de 63 valores.

### Motivo

La distancia máxima utiliza toda la extensión detectada y evita depender de un único
dedo, que podría estar flexionado. La reflexión permite que una pose equivalente de
mano izquierda o derecha comparta representación. Mantener todas las coordenadas
conserva la geometría disponible sin agregar todavía un clasificador o plantillas.

### Consecuencias

- Los vectores son invariantes a traslación y escala uniforme.
- Los valores no finitos, formas distintas de `(21, 3)` y manos sin extensión se
  rechazan antes de comparar.
- La reflexión puede desactivarse desde la configuración cuando la lateralidad tenga
  significado para una seña.
- Cada vector representa una mano por separado. La construcción de plantillas de dos
  manos deberá definir un orden canónico y características espaciales entre manos.

## 2026-07-31 — Orden canónico y formato persistido de plantillas

### Problema

MediaPipe no garantiza que dos manos lleguen en el mismo orden entre imágenes. La
concatenación directa produciría vectores distintos para la misma seña y la
normalización individual eliminaría la posición relativa entre manos. También se
necesita relacionar de forma verificable cada fila persistida con su imagen fuente.

### Alternativas consideradas

- Conservar el orden de detección de MediaPipe.
- Ordenar por la coordenada horizontal de las muñecas.
- Ordenar por lateralidad y rechazar pares ambiguos.
- Guardar un archivo por muestra o matrices agrupadas por clase.

### Decisión

Para dos manos, exigir exactamente una lateralidad `Left` y una `Right`, y
concatenar siempre Left antes de Right. Después de los dos vectores de 63 valores,
agregar el desplazamiento `Right wrist - Left wrist` dividido por la escala media
de ambas manos. El vector de dos manos tiene por tanto 129 valores.

Cuando una carpeta mezcla muestras de una y dos manos, conservar la cantidad con
mayoría única y rechazar las inconsistentes; un empate invalida la clase. Persistir
una matriz `float32` por clase en un NPZ comprimido y un JSON versionado con
metadatos, lateralidades, umbrales y rutas relativas de las muestras.

### Motivo

La lateralidad expresa una identidad estable que no depende de la posición de las
manos en la imagen. El desplazamiento relativo conserva dirección y separación de
las muñecas, manteniendo invariancia a traslación y escala uniforme. Agrupar varias
muestras por clase permite compararlas todas sin reconstruir plantillas al iniciar.

### Consecuencias

- El orden del resultado no depende del orden devuelto por MediaPipe.
- Imágenes de dos manos con lateralidad ausente o duplicada se rechazan.
- Las plantillas de una mano tienen dimensión 63 y las de dos manos dimensión 129.
- El reconocimiento deberá comparar únicamente observaciones y plantillas con igual
  cantidad de manos y dimensión.
- Las imágenes originales no se modifican y los artefactos pueden regenerarse con
  `python scripts/build_templates.py`.

## 2026-07-31 — Estrategia de comparación y rechazo

### Problema

Cada seña dispone de varias muestras y el motor debe producir una puntuación
estable, evitar comparar cantidades distintas de manos y rechazar observaciones sin
una similitud suficiente. También debe resolver empates de forma reproducible.

### Alternativas consideradas

- Promediar primero todos los vectores de una clase.
- Promediar las similitudes de todas las muestras.
- Usar la mejor similitud entre todas las muestras de cada clase.
- Aplicar únicamente un umbral global o permitir umbrales por seña.

### Decisión

Calcular similitud coseno contra todas las muestras compatibles y utilizar la mejor
puntuación de cada seña. Ordenar por puntuación descendente y por `gesture_id`
ascendente en caso de empate. Inferir una mano para vectores de dimensión 63 y dos
manos para dimensión 129, sin comparar otras dimensiones.

La precedencia de umbrales es: sobrescritura entregada a `GestureMatcher`, umbral
individual persistido y umbral global. Si la mejor puntuación no alcanza el umbral,
devolver `MatchResult` desconocido conservando esa puntuación. Entradas vacías,
no finitas, de norma cero o de dimensión incompatible devuelven puntuación `0.0`.

### Motivo

La mejor muestra conserva variaciones válidas de orientación y apertura sin diluirlas
en un promedio que podría no representar ninguna pose real. El filtro por dimensión
impide comparar accidentalmente una observación de una mano con una plantilla de
dos. El desempate explícito hace las pruebas y resultados reproducibles.

### Consecuencias

- Agregar muestras amplía la cobertura de una clase sin reconstruir un centroide.
- Una muestra atípica podría elevar la similitud; los umbrales por seña permiten
  ajustarlo y deberán calibrarse con ejemplos negativos.
- Las puntuaciones de cada muestra se registran únicamente en nivel `DEBUG`.
- El motor permanece separado del video hasta que exista estabilización temporal.

## 2026-07-31 — Estrategia de estabilización temporal

### Problema

Una coincidencia válida en un solo fotograma puede desaparecer o alternar por
pequeñas variaciones de landmarks. Confirmar únicamente el último resultado haría
parpadear la etiqueta y la imagen de presentación.

### Alternativas consideradas

- Exigir solo una racha de resultados consecutivos.
- Elegir únicamente la mayoría dentro de una ventana.
- Suavizar las puntuaciones numéricas de cada clase.
- Combinar dominancia, racha mínima, retención e histéresis con estado explícito.

### Decisión

Mantener una ventana acotada de `MatchResult` y contar solo resultados aceptados.
Una nueva seña requiere una mayoría única, al menos `stable_frames` votos y una
racha final de `min_consecutive_frames`. Los empates no confirman ni cambian una
seña.

Después de confirmar, conservar el último resultado durante `hold_frames` ausencias.
Aplicar histéresis reduciendo en `hysteresis_frames` el número de votos necesario
para mantener una seña ya activa, sin reducir el umbral para activar una nueva.
Una candidata diferente debe cumplir de nuevo dominancia, votos y consecutividad.

### Motivo

La mayoría tolera interrupciones aisladas, la racha evita confirmar ruido disperso,
la retención cubre pérdidas breves y la histéresis evita alternancias cerca del
umbral. El filtro devuelve el `MatchResult` confirmado completo, por lo que también
mantiene estable la ruta de la imagen asociada.

### Consecuencias

- La latencia de confirmación y permanencia se controla desde `config/config.yaml`.
- `reset()` obliga a confirmar nuevamente después de reiniciar el flujo.
- El filtro no mezcla cantidades de manos; recibe resultados ya validados por
  `GestureMatcher`.
- Los valores expresados en fotogramas deberán calibrarse con el FPS observado al
  integrar el motor al video.

## 2026-07-31 — Panel lateral y caché de imágenes de presentación

### Problema

La interfaz debe mostrar video, reconocimiento e imagen relacionada sin deformar
recursos ni introducir una lectura de disco en cada fotograma. Además, una imagen
no debe aparecer antes de que la seña sea confirmada temporalmente.

### Alternativas consideradas

- Superponer la imagen directamente sobre el video.
- Crear un panel lateral de tamaño estable.
- Precargar todas las imágenes al iniciar.
- Cargar cada imagen al primer uso y conservar éxitos y fallos en caché.
- Usar imágenes de referencia cuando falte una imagen de presentación.

### Decisión

Componer una sola matriz OpenCV con video a la izquierda y panel de resultado a la
derecha. El panel recibe únicamente el `MatchResult` devuelto por `TemporalFilter`.
Solo consulta la imagen cuando `accepted=True` y existe `display_image_path`.

Usar `ImageCache` con carga diferida por ruta relativa al proyecto. Guardar en caché
tanto imágenes válidas como fallos para no repetir accesos. Ajustar cada imagen al
rectángulo configurado conservando su relación de aspecto y rellenando el espacio
restante, sin recortar ni estirar. Usar exclusivamente `assets/display_images/` y
nunca las referencias como sustituto.

### Motivo

El panel evita cubrir landmarks y mantiene separada la información de resultado.
La carga diferida evita memoria innecesaria para señas nunca mostradas, mientras la
caché elimina I/O repetido. La condición `accepted=True` hace que la imagen siga la
misma confirmación e histéresis que la etiqueta.

### Consecuencias

- El primer uso de una seña puede realizar una única lectura de disco.
- Una imagen ausente se informa y no se intenta cargar nuevamente por fotograma.
- Agregar una imagen de presentación requiere reconstruir metadatos.
- El tamaño final se controla con `display.result_image_width` y
  `display.result_image_height`.
- La validación visual y el impacto real en FPS quedan pendientes en cámara física.
