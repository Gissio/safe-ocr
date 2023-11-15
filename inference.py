# OCR para SAFE (telegramas y certificados)
# Inferencia
#
# por Gissio
# MIT License

from PIL import Image

import numpy as np

import torch
from torch import nn


# Definición de red

class SAFENetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 * 6 * 512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2_1 = nn.Sequential(
            nn.Linear(4096, 10))
        self.fc2_2 = nn.Sequential(
            nn.Linear(4096, 10))
        self.fc2_3 = nn.Sequential(
            nn.Linear(4096, 10))

        return

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out1 = self.fc2_1(out)
        out2 = self.fc2_2(out)
        out3 = self.fc2_3(out)

        return out1, out2, out3


def ocr(image_paths):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    # Posición de los valores a recuperar

    crop_tab_x = np.array([1195, 1410])
    crop_tab_y = np.array(
        [742, 815, 887, 957, 1029, 1099, 1170, 1241, 1314, 1386, 1455, 1538]) + 3

    crop_width = 220
    crop_height = 85

    model_mean = 39  # De: np.mean(self.images)
    model_std = 87  # De: np.std(self.images)

    # Inicialización de la red

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SAFENetwork().to(device)
    model.load_state_dict(torch.load('model.pt'))

    outputs = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("L")

        output = []

        index = 0

        for crop_y in crop_tab_y:
            for crop_x in crop_tab_x:
                # Recorta la imagen

                cropped_image = image.crop((crop_x,
                                            crop_y,
                                            crop_x + crop_width,
                                            crop_y + crop_height))

                # Test:
                # cropped_image.save(f'{index}_out.png')

                # Invierte los colores y normaliza

                np_image = 255 - np.array(cropped_image, dtype='uint8')
                np_image = (np_image - model_mean) / model_std

                # Lo pasamos a pytorch

                torch_image = torch.tensor(np_image).float()
                torch_image = torch_image.unsqueeze(0).unsqueeze(0)

                # Evalúa la imagen

                output1, output2, output3 = model(torch_image.to(device))

                # Respuesta

                _, predicted1 = torch.max(output1.data, 1)
                _, predicted2 = torch.max(output2.data, 1)
                _, predicted3 = torch.max(output3.data, 1)
                predicted = 100 * predicted1 + 10 * predicted2 + predicted3

                output.append(predicted.item())

                index += 1

        outputs.append(output)

    return outputs


result = ocr('img/0211900018X.png')
print(result)
