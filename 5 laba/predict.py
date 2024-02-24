import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv('avocado.csv')
df = df.drop(columns = ['Unnamed: 0'],axis=1)
df = df.dropna()
df['type'] = df['type'].replace({'conventional': 1, 'organic': 2})

region_mapping = {
    'Albany': 1,
    'Sacramento': 2,
    'Northeast': 3,
    'NorthernNewEngland': 4,
    'Orlando': 5,
    'Philadelphia': 6,
    'PhoenixTucson': 7,
    'Pittsburgh': 8,
    'Plains': 9,
    'Portland': 10,
    'RaleighGreensboro': 11,
    'RichmondNorfolk': 12,
    'Roanoke': 13,
    'SanDiego': 14,
    'Atlanta': 15,
    'SanFrancisco': 16,
    'Seattle': 17,
    'SouthCarolina': 18,
    'SouthCentral': 19,
    'Southeast': 20,
    'Spokane': 21,
    'StLouis': 22,
    'Syracuse': 23,
    'Tampa': 24,
    'TotalUS': 25,
    'West': 26,
    'NewYork': 27,
    'NewOrleansMobile': 28,
    'Nashville': 29,
    'Midsouth': 30,
    'BaltimoreWashington': 31,
    'Boise': 32,
    'Boston': 33,
    'BuffaloRochester': 34,
    'California': 35,
    'Charlotte': 36,
    'Chicago': 37,
    'CincinnatiDayton': 38,
    'Columbus': 39,
    'DallasFtWorth': 40,
    'Denver': 41,
    'Detroit': 42,
    'GrandRapids': 43,
    'GreatLakes': 44,
    'HarrisburgScranton': 45,
    'HartfordSpringfield': 46,
    'Houston': 47,
    'Indianapolis': 48,
    'Jacksonville': 49,
    'LasVegas': 50,
    'LosAngeles': 51,
    'Louisville': 52,
    'MiamiFtLauderdale': 53,
    'WestTexNewMexico': 54
}

df['region'] = df['region'].map(region_mapping)

df['month'] = df['Date'].apply(lambda x: int(x.split('-')[1]))
df['day'] = df['Date'].apply(lambda x: int(x.split('-')[2]))

df = df.drop(columns = ['Date'],axis=1)
cols = [
    'Total Volume',
    '4046',
    '4225',
    '4770',
    'Total Bags',
    'Small Bags', 
    'Large Bags',
    'XLarge Bags',
    ]

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

le = LabelEncoder()
df['AveragePrice'] = le.fit_transform(df['AveragePrice'])

train_data = df[df['year'] < 2018]
test_data = df[df['year'] == 2018]

x_train = train_data.drop('AveragePrice', axis=1)
y_train = train_data['AveragePrice']

x_test = test_data.drop('AveragePrice', axis=1)
y_test = test_data['AveragePrice']

model_names = [
    'Linear Regression',
    'Decision Tree Regression',
    'Random Forest Regression',
    'Gradient Boosting Regression',
    ]

models = [
    LinearRegression(),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(n_estimators=200, max_features=0.35, random_state=42),
    GradientBoostingRegressor(random_state=42),
    ]


for model, name in zip(models, model_names):
    print(f"Обучение {name}...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    print(f"{name} обучение закочено. Mean Squared Error: {mse}. Mean Absolute Error: {mae}. R2: {r2},\n")