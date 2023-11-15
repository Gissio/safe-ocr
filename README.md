# OCR para telegramas y certificados

Instalar las dependencias:

    pip install -r requirements.txt

## Construcción del dataset

* Descargar telegramas.
* Descargar datos en formato .csv de https://www.argentina.gob.ar/dine/resultados-electorales/elecciones-2023.
* Ejecutar:

    python build_dataset.py

* Se genera: `generales.pickle` (caché), `generales_02_images.bin` con las imágenes recortadas (16 GB) y `generales_02_labels.bin` con las etiquetas.

## Entrenamiento

* Ejecutar:

    python train.py

## Inferencia

* Ejecutar:

    python inference.py
