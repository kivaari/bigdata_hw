import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class DecisionTree_Classifier:
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        # Количество уникальных классов в целевой переменной
        self.n_classes = len(np.unique(y))
        # Количество признаков
        self.n_features = X.shape[1]
        # Строим дерево решений
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Количество образцов каждого класса
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        # Предполагаемый класс - класс с наибольшим количеством образцов
        predicted_class = np.argmax(n_samples_per_class)
        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            # Выбор лучшего разделения
            feature, threshold = self._best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature'] = feature
                node['threshold'] = threshold
                # Рекурсивное построение дерева для левого и правого поддеревьев
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        # Находим наилучшее разделение по критерию Джини
        best_gini = 1
        best_feature, best_threshold = None, None
        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._gini_impurity(X, y, feature, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_impurity(self, X, y, feature, threshold):
        # Вычисление критерия Джини для разделения
        indices_left = X[:, feature] < threshold
        y_left = y[indices_left]
        y_right = y[~indices_left]
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        # Расчет критерия Джини для левой и правой подгрупп
        gini_left = 0 if n_left == 0 else 1 - sum((np.sum(y_left == c) / n_left) ** 2 for c in range(self.n_classes))
        gini_right = 0 if n_right == 0 else 1 - sum((np.sum(y_right == c) / n_right) ** 2 for c in range(self.n_classes))
        
        # Общий критерий Джини для разделения
        if n_total == 0:
            return 0
        
        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return gini

    def _predict(self, sample, tree):
        # Рекурсивное предсказание класса для образца
        if 'predicted_class' in tree:
            return tree['predicted_class']
        else:
            feature = tree['feature']
            threshold = tree['threshold']
            if sample[feature] < threshold:
                return self._predict(sample, tree['left'])
            else:
                return self._predict(sample, tree['right'])

    def predict(self, X):
        # Предсказание классов для всех образцов в X
        return [self._predict(sample, self.tree) for sample in X]


def main():
    # Колонка, которую будем предсказывать
    TARGET = "Класс"

    # Создаем датафрейм путем открытия файла
    df = pd.read_csv("shrooms.txt", encoding="Windows-1251", sep="\t")
    # Удаляем строки, в которых пропущены какие-либо значения
    df.dropna(inplace=True)

    # убираем оповещение об изменении данных с помощью метода replace
    pd.set_option('future.no_silent_downcasting', True)
    # Словарь с значениями данных, которые нужно заменить
    class_mapping = {'съедобный': 1, 'ядовитый': 0}
    # Заменяем все значения в столбце по ключу в словаре
    df['Класс'] = df['Класс'].replace(class_mapping).infer_objects(copy=False)

    # Создает список колонок из датафрейма, тип данных которых = object
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Вызов энкодера 
    encoder = OneHotEncoder()
    # Энкодер нужен, чтобы закодировать категориальные данные в колонках
    encoded_data = encoder.fit_transform(df[cat_cols])
    # Создание нового датафрейма с закодированными в колонках данными
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(cat_cols))
    df_encoded = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)

    X = df_encoded.drop(TARGET, axis=1)
    y = df[TARGET]

    # Разделяем данные на тренировочную и тестовую выборки
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=42)

    # Вызываем класс дерева решений
    clf = DecisionTree_Classifier(max_depth=1, random_state=42)
    # Вызываем метод fit из класса дерева решений для сбора значений из обучающей выборки
    clf.fit(train_X.values, train_y.values)
    # Вызываем метод predict для предсказания значений тестовой выборки
    result = clf.predict(test_X.values)

    dtc = DecisionTreeClassifier(
        criterion='entropy',
        splitter='best',
        max_depth=5
    )
    dtc.fit(train_X, train_y)
    result_2 = dtc.predict(test_X)

    # Подсчитываем метрики модели
    f1 = round(f1_score(test_y, result), 2)
    accuracy = round(accuracy_score(test_y, result), 2)
    precision = round(precision_score(test_y, result), 2)
    recall = round(recall_score(test_y, result), 2)

    # Подсчитываем метрики модели
    f1dtc = f1_score(test_y, result_2)
    accuracydtc = accuracy_score(test_y, result_2)
    precisiondtc = precision_score(test_y, result_2)
    recalldtc = recall_score(test_y, result_2)

    print("Sci-kit learn dtc:")
    print(f"F1 Score: {f1dtc}\nAccuracy Score: {accuracydtc}\nPrecision Score: {precisiondtc}\nRecall Score: {recalldtc}")
    print("My dtc:")
    print(f"F1 Score: {f1}\nAccuracy Score: {accuracy}\nPrecision Score: {precision}\nRecall Score: {recall}")
    print("Метки классов модели:", clf.tree['predicted_class'])
    print("Результаты предсказания:")
    for i in range(20):
        predicted_class = result[i]
        actual_class = test_y.iloc[i]
        print(f"Предсказанное значение: {predicted_class}, Фактическое значение: {actual_class}")


if __name__ == "__main__":
    main()
