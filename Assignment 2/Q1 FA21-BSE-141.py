# 27th March, 2024
# CSC354 – Assignment1 – ML – Concept Learning
# Muhammad Ammar Mukhtar
# FA21-BSE-141


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("datasaurus.csv")

print(dataset.head())

X = dataset.drop('dataset', axis=1)
y = dataset['dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

j48_baseline = DecisionTreeClassifier(random_state=42)
rf_baseline = RandomForestClassifier(random_state=42)

j48_baseline.fit(X_train, y_train)
rf_baseline.fit(X_train, y_train)

j48_pred_baseline = j48_baseline.predict(X_test)
rf_pred_baseline = rf_baseline.predict(X_test)

j48_accuracy_baseline = accuracy_score(y_test, j48_pred_baseline)
rf_accuracy_baseline = accuracy_score(y_test, rf_pred_baseline)

print("Baseline J48 Accuracy:", j48_accuracy_baseline)
print("Baseline Random Forest Accuracy:", rf_accuracy_baseline)

j48_param_grid = {
    'criterion': ['gini'],
    'max_depth': [None, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

j48_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), j48_param_grid, cv=5)
j48_grid_search.fit(X_train, y_train)

print("Best parameters for J48:", j48_grid_search.best_params_)
print("Best score for J48:", j48_grid_search.best_score_)

rf_param_dist = {
    'n_estimators': [50],
    'max_features': ['auto'],
    'max_depth': [80] + [None],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
    'bootstrap': [True]
}
rf_random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_dist, n_iter=1, cv=5, random_state=42)
rf_random_search.fit(X_train, y_train)


print("Best parameters for Random Forest:", rf_random_search.best_params_)
print("Best score for Random Forest:", rf_random_search.best_score_)
