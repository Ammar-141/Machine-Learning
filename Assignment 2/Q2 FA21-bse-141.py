# 27th March 2024
# CSC354 – Assignment1 – ML – Concept Learning
# Muhammad Ammar Mukhtar
# FA21-BSE-141


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

car_df = pd.read_csv("cars-dataset.csv")

print(car_df.head())

car_df_encoded = pd.get_dummies(car_df, drop_first=True)

X = car_df_encoded.drop('selling_price', axis=1)
y = car_df_encoded['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_reg_baseline = DecisionTreeRegressor(random_state=42)

dt_reg_baseline.fit(X_train, y_train)

y_pred_baseline = dt_reg_baseline.predict(X_test)

rmse_baseline = mean_squared_error(y_test, y_pred_baseline, squared=False)
print("Baseline RMSE:", rmse_baseline)

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

