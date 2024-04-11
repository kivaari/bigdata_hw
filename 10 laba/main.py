import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df_train = pd.read_csv('/content/train.csv')

df_train

df_train.info()

df_train['CITY'] = df_train['ADDRESS'].str.split(',').str.get(1)
df_train['CITY'].value_counts()

cat_features_mask = (df_train.dtypes == "object").values

num_columns = df_train.columns[~cat_features_mask]
scaler = StandardScaler()
for col in num_columns:
    df_train[col] = scaler.fit_transform(df_train[col].values.reshape(-1, 1))
cat_columns = df_train.columns[cat_features_mask]
label_encoder = LabelEncoder()
for col in cat_columns:
    df_train[col] = label_encoder.fit_transform(df_train[col])

df_train

x = df_train.drop(["TARGET(PRICE_IN_LACS)"], axis=1)
y = df_train["TARGET(PRICE_IN_LACS)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)
gbr_score = gbr.score(x_test, y_test)
gbr_score

dtr = DecisionTreeRegressor(max_depth = 5)
dtr.fit(x_train, y_train)
dtr_score = dtr.score(x_test, y_test)
dtr_score