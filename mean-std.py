# OCR para SAFE (Sistema anti-fraude electoral)
# Cálculo de media y varianza del dataset
#
# por Gissio
# MIT License

import numpy as np

img_size = (220, 85)
img_pixel_num = img_size[0] * img_size[1]

dataset = np.fromfile(file='generales_02_images.bin', 
                      dtype='uint8',
                      offset=0,
                      count=100000 * img_pixel_num)

print(np.mean(dataset))
print(np.std(dataset))
print(dataset.shape)
