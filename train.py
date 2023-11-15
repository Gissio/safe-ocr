# OCR para SAFE (Sistema anti-fraude electoral)
# Entrenamiento
#
# por Gissio
# MIT License

import os

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler


class SAFEDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path):
        # Load images as memory map

        self.images = np.memmap(
            images_path, dtype='uint8', mode='r').__array__()
        self.labels = np.memmap(
            labels_path, dtype='uint32', mode='r').__array__()

        img_width = 220
        img_height = 85

        self.images = np.reshape(
            self.images, (self.labels.shape[0], img_height, img_width))

        # Normalización con media y desvío estándar
        self.mean = 39  # De: np.mean(self.images)
        self.std = 87  # De: np.std(self.images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        normalized_image = (image - self.mean) / self.std

        torch_image = torch.tensor(normalized_image).float()
        torch_image = torch_image[None, :]

        label = self.labels[index]
        label_1 = int(label / 100) % 10
        label_2 = int(label / 10) % 10
        label_3 = int(label % 10)

        torch_label = torch.tensor([label_1, label_2, label_3])
        # torch_label = torch.tensor(label_3)

        return torch_image, torch_label


def init_weights(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)

        model.bias.data.fill_(0.01)


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


def plot_graphs(accuracy_history, train_loss_history, validation_loss_history):
    plt.figure()
    plt.plot(np.arange(len(accuracy_history)) + 1, accuracy_history)
    plt.title('Train accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.savefig('graph_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(np.arange(len(train_loss_history)) +
             1, train_loss_history, label='Train')
    plt.plot(np.arange(len(validation_loss_history)) + 1,
             validation_loss_history, label='Validation')
    plt.title('Train and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('graph_loss.png')
    plt.close()


# Inicialización

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Carga de datos

batch_size = 64

dataset = SAFEDataset('generales_02_images.bin',
                      'generales_02_labels.bin')
dataloader = torch.utils.data.DataLoader(dataset)

train_num = 856928 # En total hay 896928 ejemplos
validation_num = 20000
test_num = 20000

indices = list(range(train_num + validation_num + test_num))

np.random.seed(142857)
np.random.shuffle(indices)

train_sampler = SubsetRandomSampler(indices[:train_num])
validation_sampler = SubsetRandomSampler(
    indices[train_num:train_num + validation_num])
test_sampler = SubsetRandomSampler(
    indices[train_num + validation_num:train_num + validation_num + test_num])

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                sampler=validation_sampler)
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          sampler=test_sampler)

# Inicialización del modelo

model = SAFENetwork().to(device)

# if not os.path.exists('model.pt'):
#     model.apply(init_weights)
# else:
#     model.load_state_dict(torch.load('model.pt'))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.005,
                            weight_decay=0.005,
                            momentum=0.9)

# Entrenamiento y validación

epochs_num = 10000

train_loss_history = []
validation_loss_history = []
validation_accuracy_history = []

total_step = len(train_loader)

for epoch in range(epochs_num):
    print('Training', end='', flush=True)

    # Entrenamiento

    for i, (images, labels) in enumerate(train_loader):
        # Copia datos a la GPU
        images = images.to(device)
        labels = labels.to(device)

        # Paso forward
        outputs1, outputs2, outputs3 = model(images)
        loss = criterion(outputs1, labels[:, 0]) +\
            criterion(outputs2, labels[:, 1]) +\
            criterion(outputs3, labels[:, 2])

        # Paso backward y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Consola
        if i % 10 == 0:
            current = (i + 1) * len(images)
            size = len(train_loader)

            print('.', end='', flush=True)

    train_loss = loss.item()
    train_loss_history.append(train_loss)

    print()

    print(f'Train: epoch {epoch + 1}/{epochs_num}, loss {train_loss:.4f}')

    # Validación

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs1, outputs2, outputs3 = model(images)
            loss = criterion(outputs1, labels[:, 0]) +\
                criterion(outputs2, labels[:, 1]) +\
                criterion(outputs3, labels[:, 2])

            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            _, predicted3 = torch.max(outputs3.data, 1)
            predicted = 100 * predicted1 + 10 * predicted2 + predicted3
            labels_value = 100 * labels[:, 0] + 10 * labels[:, 1] + labels[:, 2]

            correct += (predicted == labels_value).sum().item()
            total += labels.size(0)

            del images, labels, outputs1, outputs2, outputs3

    validation_loss = loss.item()
    validation_loss_history.append(validation_loss)

    validation_accuracy = 100 * correct / total
    validation_accuracy_history.append(validation_accuracy)

    print(
        f'Validation: accuracy {(validation_accuracy):>0.2f}%, loss {validation_loss:.4f}')

    plot_graphs(validation_accuracy_history,
                train_loss_history, validation_loss_history)

    os.makedirs('model2', exist_ok=True)
    torch.save(model.state_dict(), f'model2/model_{epoch + 1}.pt')

    print('----------------')

# Test

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs1, outputs2, outputs3 = model(images)
        loss = criterion(outputs1, labels[:, 0]) +\
            criterion(outputs2, labels[:, 1]) +\
            criterion(outputs3, labels[:, 2])

        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        _, predicted3 = torch.max(outputs3.data, 1)
        predicted = 100 * predicted1 + 10 * predicted2 + predicted3
        labels_value = 100 * labels[:, 0] + 10 * labels[:, 1] + labels[:, 2]

        correct += (predicted == labels_value).sum().item()
        total += labels.size(0)

        del images, labels, outputs1, outputs2, outputs3

accuracy = correct / total

print(f'Test: accuracy {(100 * accuracy):>0.2f}%')

print('Done.')

exit()
