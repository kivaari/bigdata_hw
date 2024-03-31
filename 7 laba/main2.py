import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Read the data file
df = pd.read_csv("audi.csv")
# Check data types in the columns of the table.
print(df.info())

# Drop rows with missing values
df.dropna(inplace=True)

cat_cols = [
    "model",
    "transmission",
    "fuelType",
]

num_cols = [
    "year",
    "price",
    "mileage",
    "tax",
    "mpg",
    "engineSize",
]

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

print(df)

# Разбиваем данные на обучающую и тестовую выборки в соотношении 80/20
x = df.drop(['price'], axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Регрессионное дерево решений
dtr = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)

# Предсказание
y_pred = dtr.predict(x_test)

# Выводим скор моделей с помощью встроенного метода score
print("Descision Tree Regressor score:", dtr.score(x_test, y_pred))

# График предсказания регрессионного дерева решений
plt.scatter(y_test, y_pred, alpha=0.2)
plt.plot([0, max(y_test)], [0, max(y_pred)]) 
plt.xlabel('Настоящая цена')
plt.ylabel('Предсказанная цена DTR')
plt.title('Сравнение фактических и предсказанных значений')

plt.show()
