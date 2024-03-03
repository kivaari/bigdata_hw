# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Достаем данные из файла
df = pd.read_csv('fixed_values_ds.csv')

# Убираем строки в которых есть пропущенные значения
df.dropna(inplace=True)

# Разделяем датасет на обучающую и тестовую выборки в соотношении 80/20
x = df.drop('result', axis=1)
y = df['result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Обучаем дерево решений
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# Предсказываем значения по тестовой выборке
y_pred = dtc.predict(x_test)

# Считаем скор модели сравнивая настоящие значения и предсказанные
f1 = round(f1_score(y_test, y_pred), 2)

# Считаем долю правильных ответов модели на офне всех ответов
acc = round(accuracy_score(y_test, y_pred), 2)

# Для оценки качества работы модели на каждом из классов по отдельности используем метрики precision (точность) и recall (полнота).

# Precision можно интерпретировать как долю объектов, названных классификатором положительными и при этом действительно являющимися положительными
precision = round(precision_score(y_test, y_pred), 2)

# Recall показывает, какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм.
recall = round(recall_score(y_test, y_pred), 2)

# Выводим все метрики
print(f"F1-score: {f1} \nAccuracy score: {acc} \nPrecision: {precision} \nRecall: {recall}")