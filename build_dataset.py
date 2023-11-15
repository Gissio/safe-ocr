# OCR para SAFE (telegramas y certificados)
# Preparación del dataset para entrenamiento
#
# por Gissio
# MIT License

import os
from PIL import Image
import pandas as pd
import numpy as np

# Datos de labels

if not os.path.exists('generales.pickle'):
    # Bajado de https://www.argentina.gob.ar/dine/resultados-electorales/elecciones-2023
    csv_path = '../../Datasets/2023 Generales Datos/ResultadosElectorales_2023.csv'

    csv_dtype = {
        "año": int,
        "eleccion_tipo" : str,
        "recuento_tipo" : str,
        "padron_tipo" : str,
        "distrito_id" : int,
        "distrito_nombre" : str,
        "seccionprovincial_id" : int,
        "seccionprovincial_nombre" : str,
        "seccion_id" : int,
        "seccion_nombre" : str,
        "circuito_id" : str,
        "circuito_nombre" : str,
        "mesa_id" : int,
        "mesa_tipo" : str,
        "mesa_electores" : int,
        "cargo_id" : int,
        "cargo_nombre" : str,
        "agrupacion_id" : int,
        "agrupacion_nombre" : str,
        "lista_numero" : str,
        "lista_nombre" : str,
        "votos_tipo" : str,
        "votos_cantidad": int
    }

    df = pd.read_csv(csv_path, dtype=csv_dtype)
    df.to_pickle('generales.pickle')

else:
    df = pd.read_pickle('generales.pickle')

# Importación

def prepare_dataset(file_names, dataset_name):
    y_from_agrupacion = {
        132: 0,
        133: 1,
        134: 2,
        135: 3,
        136: 4,
    }

    y_from_votostipo = {
        'NULO': 6,
        'RECURRIDO': 7,
        'IMPUGNADO': 8,
        'COMANDO': 9,
        'EN BLANCO': 10
    }

    crop_tab_x = [1195, 1410]
    crop_tab_y = [742, 815, 887, 957, 1029, 1099, 1170, 1241, 1314, 1386, 1455, 1538]

    crop_width = 220
    crop_height = 85

    file_num = len(file_names)

    records_per_file = len(crop_tab_x) * len(crop_tab_y)
    record_num = file_num * records_per_file

    images = open(dataset_name + '_images.bin', 'wb')
    labels = open(dataset_name + '_labels.bin', 'wb')

    index = 0

    for file_index, file_name in enumerate(file_names):
        file_path = dir_path + '/' + file_name

        print(f'Processing image {file_index}: {file_path}...')

        # Datos

        distrito = int(file_name[0:2])
        seccion = int(file_name[2:5])
        mesa = int(file_name[5:10])

        datos_de_mesa = df[(df.distrito_id == distrito) &
                        (df.seccion_id == seccion) &
                        (df.mesa_id == mesa) &
                        ((df.cargo_id == 1) | (df.cargo_id == 8))]

        file_labels = np.zeros((12, 2), dtype='uint32')

        for row_index, row in datos_de_mesa.iterrows():
            x = 1 if row.cargo_id == 8 else 0
            if row.votos_tipo == 'POSITIVO':
                y = y_from_agrupacion[row.agrupacion_id]

                file_labels[y][x] = row.votos_cantidad
                file_labels[5][x] += row.votos_cantidad
                file_labels[11][x] += row.votos_cantidad

            else:
                y = y_from_votostipo[row.votos_tipo]

                file_labels[y][x] = row.votos_cantidad
                file_labels[11][x] += row.votos_cantidad

        # Imágenes

        file_image = Image.open(file_path).convert("L")

        cropped_data = None

        for y, crop_y in enumerate(crop_tab_y):
            crop_y += 3 # Adjustment

            for x, crop_x in enumerate(crop_tab_x):
                cropped_image = file_image.crop((crop_x,
                                            crop_y,
                                            crop_x + crop_width,
                                            crop_y + crop_height))

                # Test:
                # cropped_image.save(f'{index}_out.png')
                
                image = 255 - np.array(cropped_image, dtype='uint8')
                images.write(image.tobytes())

                label = file_labels[y][x]
                labels.write(label.tobytes())

                index += 1

    return

# `dir_path` contiene los telegramas bajados de https://mega.nz
#
# Formato de los archivos: aabbbcccccX.jpg
# * aa: código de distrito
# * bbb: código de sección
# * ccccc: código de mesa
# 
# Ejemplo: 0200100001X.jpg
dir_path = '../../Datasets/2023 Generales Telegramas alineados/02'

file_names = os.listdir(dir_path)

prepare_dataset(file_names[:], 'generales_02')
