# California Housing Price Prediction

A regression project predicting median house prices using the California Housing dataset, built end-to-end with scikit-learn Pipelines and hyperparameter tuning.

## Dataset
- Source: `sklearn.datasets.fetch_california_housing`
- 20,640 rows, 8 numeric features (median income, house age, rooms, population, location, etc.)
- Target: median house value (in $100,000s)

## Approach
1. **EDA** — checked nulls, duplicates, and correlation of each feature with house price
2. **Preprocessing** — StandardScaler wrapped in a Pipeline (no missing values or categorical columns in this dataset)
3. **Baseline models** — Linear Regression vs Random Forest Regressor
4. **Evaluation** — MAE, RMSE, R² on a held-out test set
5. **Hyperparameter tuning** — GridSearchCV (5-fold CV) on `max_depth` and `n_estimators`
6. **Feature importance** — extracted from the tuned Random Forest

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.533 | 0.746 | 0.576 |
| Random Forest (tuned) | 0.327 | 0.505 | **0.805** |

Random Forest outperformed Linear Regression by a wide margin, reducing average error from ~$53k to ~$33k.

## Key finding
`MedInc` (median income) was the strongest predictor by far (52.6% feature importance), consistent with the correlation seen during EDA. Location (`Latitude`/`Longitude`) was the next most important factor.

## Tools
Python, pandas, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV), matplotlib/seaborn
