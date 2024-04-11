# Импортируем библиотеки
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Если доступна видеокарта то используем куда ядра для ускорения обучения
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Открываем наши данные
df0 = pd.read_csv('/content/0.csv', header = None)
df1 = pd.read_csv('/content/1.csv', header = None)
df2 = pd.read_csv('/content/2.csv', header = None)
df3 = pd.read_csv('/content/3.csv', header = None)

# Объединяем все данные в один датасет
df = pd.concat([df0, df1, df2, df3])

# Смотрим на наш датасет
df

# Смотрим на типы данных в колонках
df.info()

# Делим датасет на тестовую и обучающую выборки 50/50
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Преобразование в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Смотрим на получившийся тензор
X_test_tensor

# Создаем лоадеры для загрузки данных в модель
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=64)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# Пишем модельку
class GestureClassifier(nn.Module):
    # Указываем в аругентах функции размер входных данных и количество классов
    def __init__(self, input_size, num_classes):
        # Инициализируем модель
        super(GestureClassifier, self).__init__()
        # Определяем первый полносвязный слой, который принимает входные данные размером input_size и выдает 512 выходных нейронов
        self.fc1 = nn.Linear(input_size, 512)
        # Добавляем слой Batch Normalization после первого полносвязного слоя
        self.bn1 = nn.BatchNorm1d(512)
        # Определяем функцию активации ReLU для первого скрытого слоя
        self.relu1 = nn.ReLU()
        # Определяем второй полносвязный слой, который принимает 512 входных нейронов и выдает 256 выходных нейронов
        self.fc2 = nn.Linear(512, 256)
        # Добавляем слой Batch Normalization после второго полносвязного слоя
        self.bn2 = nn.BatchNorm1d(256)
        # Определяем функцию активации ReLU для второго скрытого слоя
        self.relu2 = nn.ReLU()
        # Определяем слой Dropout с коэффициентом отсева 0.5 для регуляризации модели
        self.dropout = nn.Dropout(0.5)
        # Определяем третий полносвязный слой, который принимает 256 входных нейронов и выдает 128 выходных нейронов
        self.fc3 = nn.Linear(256, 128)
        # Добавляем слой Batch Normalization после третьего полносвязного слоя
        self.bn3 = nn.BatchNorm1d(128)
        # Определяем функцию активации ReLU для третьего скрытого слоя
        self.relu3 = nn.ReLU()
        # Определяем четвертый полносвязный слой, который принимает 128 входных нейронов и выдает num_classes выходных нейронов, соответствующих классам
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Проход данных через первый полносвязный слой
        out = self.fc1(x)
        # Применение Batch Normalization к выходу первого слоя
        out = self.bn1(out)
        # Применение функции активации ReLU к выходу первого слоя
        out = self.relu1(out)
        # Проход данных через второй полносвязный слой
        out = self.fc2(out)
        # Применение Batch Normalization к выходу второго слоя
        out = self.bn2(out)
        # Применение функции активации ReLU к выходу второго слоя
        out = self.relu2(out)
        # Применение слоя Dropout к выходу второго слоя для регуляризации
        out = self.dropout(out)
        # Проход данных через третий полносвязный слой
        out = self.fc3(out)
        # Применение Batch Normalization к выходу третьего слоя
        out = self.bn3(out)
        # Применение функции активации ReLU к выходу третьего слоя
        out = self.relu3(out)
        # Проход данных через четвертый полносвязный слой, который выдает оценки для каждого класса
        out = self.fc4(out)
        # Возвращаем выходные данные
        return out

# Обозначаем размервходных данных равный размеру матрицы X_train
input_size = X_train.shape[1]
# Указываем количество классов
num_classes = 4
# Указываем модель с параметрами и выбераем способ обучения (процессор или видеокарта)
model = GestureClassifier(input_size, num_classes).to(device)

# Определяем функцию потерь Cross Entropy Loss, которая часто используется для задач классификации
criterion = nn.CrossEntropyLoss()

# Определяем оптимизатор SGD (Стохастический градиентный спуск), который обновляет веса модели с учетом градиента функции потерь
# lr - learning rate (скорость обучения), шаг, на который изменяются веса в каждом обновлении
optimizer1 = optim.SGD(model.parameters(), lr=1e-1)

# Определяем оптимизатор Adam, который также используется для обновления весов модели, но обычно работает лучше, чем SGD
# lr - learning rate (скорость обучения), шаг, на который изменяются веса в каждом обновлении
optimizer2 = optim.Adam(model.parameters(), lr=1e-1)

# Определяем шедулер StepLR, который уменьшает скорость обучения (learning rate) через каждые step_size эпох на gamma
# Это помогает улучшить сходимость модели и предотвратить переобучение
# Создали два шедулера для двух видов оптимизации
scheduler1 = StepLR(optimizer1, step_size=20, gamma=0.5)
scheduler2 = StepLR(optimizer2, step_size=20, gamma=0.5)

# Создаем функцию для вычисления метрик
def calculate_metrics(targets, predictions, precision=0, recall=0, f1=0):
    # Вычисляем precision (точность) с использованием функции precision_score из библиотеки scikit-learn
    # Если precision не указан явно (равен 0), используем average='macro' для вычисления средневзвешенной точности
    precision = precision_score(targets, predictions, average='macro') if precision == 0 else precision

    # Вычисляем recall (полноту) с использованием функции recall_score из библиотеки scikit-learn
    # Если recall не указан явно (равен 0), используем average='macro' для вычисления средневзвешенной полноты
    recall = recall_score(targets, predictions, average='macro') if recall == 0 else recall

    # Вычисляем F1-score с использованием функции f1_score из библиотеки scikit-learn
    # Если F1-score не указан явно (равен 0), используем average='macro' для вычисления средневзвешенного F1-score
    f1 = f1_score(targets, predictions, average='macro') if f1 == 0 else f1

    # Возвращаем словарь с вычисленными значениями точности, полноты и F1-score
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, test_loader, loss_fn):
    # Переводим модель в режим оценки (evaluation mode), чтобы отключить влияние dropout и batch normalization
    model.eval()

    # Инициализируем переменные для отслеживания общей потери, количества правильных предсказаний и общего числа элементов
    running_loss = 0.0
    correct = 0
    total = 0

    # Инициализируем списки для хранения всех предсказанных и целевых значений
    all_predictions = []
    all_targets = []

    # Выполняем оценку модели без вычисления градиентов
    with torch.no_grad():
        # Проходим по данным из test_loader
        for data, target in test_loader:
            # Перемещаем данные и метки на устройство (процессор или видеокарта), на котором выполняется обучение модели
            data, target = data.to(device), target.to(device)

            # Проходим данные через модель, чтобы получить выходные оценки
            output = model(data)

            # Вычисляем потери с использованием функции потерь
            loss = loss_fn(output, target)

            # Обновляем значение общей потери
            running_loss += loss.item()

            # Вычисляем количество правильных предсказаний, сравнивая предсказанные классы с истинными метками
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

            # Обновляем общее количество элементов
            total += target.size(0)

            # Расширяем список предсказанных и целевых значений, преобразуя их в массивы
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Вычисляем и выводим среднюю потерю, точность (accuracy), а также возвращаем все предсказанные и целевые значения
    return running_loss / len(test_loader), correct / total, all_predictions, all_targets

# Создаем функцию тренировки модели
def train(model, train_loader, test_loader, criterion, optimizer, scheduler=None, num_epochs=10):
    # Создаем списки для отслеживания значений потерь, точности и метрик F1, precision и recall на обучающем и тестовом наборах данных
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    train_f1 = []
    test_f1 = []
    train_precision = []
    test_precision = []
    train_recall = []
    test_recall = []

    # Проходим по заданному числу эпох
    for epoch in range(num_epochs):
        # Переводим модель в режим обучения (training mode)
        model.train()

        # Инициализируем переменные для отслеживания потерь, правильных предсказаний и общего числа элементов на текущей эпохе
        running_loss = 0.0
        correct = 0
        total = 0

        # Инициализируем списки для хранения всех предсказанных и целевых значений на обучающем наборе данных
        all_predictions_train = []
        all_targets_train = []

        # Проходим по данным из train_loader
        for i, (inputs, labels) in enumerate(train_loader):
            # Перемещаем данные и метки на устройство (процессор или видеокарта), на котором выполняется обучение модели
            inputs, labels = inputs.to(device), labels.to(device)

            # Обнуляем градиенты параметров модели
            optimizer.zero_grad()

            # Проходим данные через модель и получаем выходные оценки
            outputs = model(inputs.view(-1, input_size))

            # Вычисляем потери с использованием функции потерь
            loss = criterion(outputs, labels)

            # Обратное распространение ошибки и обновление весов модели
            loss.backward()
            optimizer.step()

            # Обновляем значение текущей потери
            running_loss += loss.item()

            # Вычисляем количество правильных предсказаний, сравнивая предсказанные классы с истинными метками
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Расширяем список предсказанных и целевых значений на обучающем наборе данных
            all_predictions_train.extend(predicted.cpu().numpy())
            all_targets_train.extend(labels.cpu().numpy())

        # Добавляем средние значения потерь и точности на обучающем наборе данных в соответствующие списки
        train_losses.append(running_loss / len(train_loader))
        train_accuracy.append(correct / total)

        # Вычисляем метрики F1, precision и recall на обучающем наборе данных и добавляем их в соответствующие списки
        train_metrics = calculate_metrics(all_targets_train, all_predictions_train)
        train_f1.append(train_metrics['f1'])
        train_precision.append(train_metrics['precision'])
        train_recall.append(train_metrics['recall'])

        # Выполняем оценку модели на тестовом наборе данных и добавляем значения потерь, точности и метрик в соответствующие списки
        test_loss, test_acc, all_predictions_test, all_targets_test = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        test_metrics = calculate_metrics(all_targets_test, all_predictions_test)
        test_f1.append(test_metrics['f1'])
        test_precision.append(test_metrics['precision'])
        test_recall.append(test_metrics['recall'])

        # Если шедулер определен, обновляем скорость обучения
        if scheduler is not None:
            scheduler.step()

        # Выводим текущие значения потерь, точности и метрик на обучающем и тестовом наборах данных
        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Train Loss: {train_losses[-1]:.4f} | '
              f'Train Acc: {train_accuracy[-1]:.4f} | '
              f'Train F1: {train_f1[-1]:.4f} | '
              f'Train Precision: {train_precision[-1]:.4f} | '
              f'Train Recall: {train_recall[-1]:.4f} | | |  '
              f'Test Loss: {test_losses[-1]:.4f} | '
              f'Test Acc: {test_accuracy[-1]:.4f} | '
              f'Test F1: {test_f1[-1]:.4f} | '
              f'Test Precision: {test_precision[-1]:.4f} | '
              f'Test Recall: {test_recall[-1]:.4f}')

    # Отображаем графики изменения потерь, точности, F1, precision и recall
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train losses', alpha=1, marker='o', markersize=3)
    plt.plot(test_losses, label='Test losses', alpha=1, marker='o', markersize=3)
    plt.title('Изменение Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracy, label='Train accuracy', alpha=1, marker='o', markersize=3)
    plt.plot(test_accuracy, label='Test accuracy', alpha=1, marker='o', markersize=3)
    plt.title('Изменение Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(train_f1, label='Train F1', alpha=1, marker='o', markersize=3)
    plt.plot(test_f1, label='Test F1', alpha=1, marker='o', markersize=3)
    plt.title('Изменение F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(train_precision, label='Train precision', alpha=0.5, marker='o', markersize=3)
    plt.plot(test_precision, label='Test precision', alpha=0.5, marker='o', markersize=3)
    plt.plot(train_recall, label='Train recall', alpha=0.5, marker='o', markersize=3)
    plt.plot(test_recall, label='Test recall', alpha=0.5, marker='o', markersize=3)
    plt.title('Изменение Precision и Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Обучаем модель с SGD оптимизатором
train(model, train_loader, test_loader, criterion, optimizer1, scheduler=scheduler1, num_epochs=100)

# Обучаем модель с Adam оптимизатором
train(model, train_loader, test_loader, criterion, optimizer2, scheduler=scheduler2, num_epochs=100)