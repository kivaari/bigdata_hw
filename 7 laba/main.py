import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Читаем нах файл с данными
df = pd.read_csv("audi.csv")
# Смотрим типы данных в колонках таблицы.
print(df.info())

# Для удобства создаем бинарную маску в которой находится список колонок с категориальными данными
cat_features_mask = (df.dtypes == "object").values

# Выкидываем строки в которы есть пустые значения
df.dropna(inplace=True)

# Самописный OneHot Encoding для кодирования категориальных данных
def onehot_encoding(x):
    unique_values = np.unique(x)
    unique_values.sort()
    encoding_matrix = np.zeros((len(x), len(unique_values)), dtype=int)
    for i, val in enumerate(x):
        encoding_matrix[i, np.where(unique_values == val)] = 1
    return encoding_matrix

# Самописный MinMax Scaler для кодирования числовых данных
def minmax_scale(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_scaled = (X - X_min) / X_range
    return X_scaled

# Кодируем категориальные колонки используя маску
cat_columns = df.columns[cat_features_mask]
for col in cat_columns:
    df[col] = onehot_encoding(df[col])

# Кодируем числовые колонки используя маску
num_columns = df.columns[~cat_features_mask]
for col in num_columns:
    df[col] = minmax_scale(df[col].values.reshape(-1, 1))

# Принт для проверки закодированных данных
#print(df)    

# Проверяем данные в каждой колонке на присутствие выбросов 
print(df['year'].unique())
print(df['price'].unique())
print(df['mileage'].unique()) # значения в колонках достаточно ровные, но все же можно сузить выборку
                              # по колонкам mileage и price для упрощения обучения

# Для устранения выбросов используем Z-функцию, которая усредняет значение в выбранной колоке
def detect_outliers_zscore(data, threshold=3):
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.abs(z_scores) > threshold

# Собственно чистим данные
outliers_mileage = detect_outliers_zscore(df['mileage'])
df = df[~outliers_mileage]

outliers_price = detect_outliers_zscore(df['price'])
df = df[~outliers_price]

# Разбиваем данные на обучающую и тестовую выборки в соотношении 80/20
x = df.drop(['price'], axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Так как предсказание цен это задача регрессии я попробую несколько моделей и выберу лучшую
# Линейня регрессия
lr = LinearRegression().fit(x_train, y_train)

# Регрессионное дерево решений
dtr = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)

# Предсказание
y_pred1 = lr.predict(x_test)
y_pred2 = dtr.predict(x_test)

print("Linear Regression score:", lr.score(x_test, y_pred1))
print("Descision Tree Regressor score:", dtr.score(x_test, y_pred2))

# График предсказания линейной регресси
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения', alpha=0.3)
plt.scatter(range(len(y_pred1)), y_pred1, color='red', label='Предсказанные значения после округления', alpha=0.3)
plt.xlabel('Настоящая цена')
plt.ylabel('Предсказанная цена LR')
plt.title('Сравнение фактических и предсказанных значений')

plt.show()

# График предсказания регрессионного дерева решений
plt.scatter(y_test, y_pred2)
plt.plot([0, max(y_test)], [0, max(y_pred2)]) 
plt.xlabel('Настоящая цена')
plt.ylabel('Предсказанная цена DTR')
plt.title('Сравнение фактических и предсказанных значений')

plt.show()
