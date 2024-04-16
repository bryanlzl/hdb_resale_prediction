# Preprocessing of Numerical and Categorical Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import statsmodels.api as sm
import statsmodels.formula.api as smf

import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_rows", 50)

current_year = 2024
year_cutoff = 2015


### Resale Price Prediction Model ###
new_resales = pd.read_csv("newresales_dataset.csv")
regression_columns = [
    "year",
    "town",
    "flat_type",
    "storey_range",
    "floor_area_sqm",
    "remaining_lease",
    "mrt_dist",
    "shopping_dist",
    "school_dist",
    "hawker_dist",
    "resale_price",
]
new_resales_reg = new_resales[regression_columns]
original_col = pd.get_dummies(new_resales_reg).columns
new_resales_reg = pd.get_dummies(new_resales_reg, drop_first=True)
dummy_var = set(original_col) - set(new_resales_reg.columns)
new_resales_reg.columns = new_resales_reg.columns.str.replace(" ", "_")
new_resales_reg.rename(
    columns={
        "price/sqm": "Price_per_sqm",
        "town_KALLANG/WHAMPOA": "town_KALLANG_WHAMPOA",
        "flat_type_MULTI-GENERATION": "flat_type_MULTI_GENERATION",
    },
    inplace=True,
)
new_resales_reg["year"] = new_resales_reg["year"] - current_year
x_variables = list(new_resales_reg.columns)
y_variable = "resale_price"
try:
    x_variables.remove(y_variable)
except ValueError:
    pass
resales_after_2020 = new_resales_reg[
    new_resales_reg["year"] >= (2020 - current_year)
].reset_index(drop=True)
resales_linear_model = smf.ols(
    data=resales_after_2020,
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)} + flat_type_3_ROOM:floor_area_sqm',
).fit()
joblib.dump(resales_linear_model, "resale_linear_model.joblib")


### Rental Price Prediction Model ###
rentals_dataset = pd.read_csv('rentals_dataset.csv', index_col=0)

rentals_dataset['date'] = pd.to_datetime(rentals_dataset['date'])
reference_date = datetime(2023, 12, 31)
rentals_dataset['year'] = rentals_dataset['date'].dt.year
rentals_dataset['months_since_signedrental'] = (
    (rentals_dataset['date'].dt.year - reference_date.year) * 12 +
    rentals_dataset['date'].dt.month - reference_date.month
)
reduced_df = rentals_dataset.drop(columns=['date', 'block', 'flat_type_group', 'postal', 'region', 'street_name', 'lat', 'lng',
                                           'nearest_MRT', 'nearest_intschool', 'nearest_shopping', 'nearest_hawker', 
                                           'price_sqm', 'year'])
preprocessed_df = pd.get_dummies(reduced_df, drop_first=True, dtype=int)
preprocessed_df.columns = preprocessed_df.columns.str.replace(' ', '_')
preprocessed_df.rename(columns={'town_KALLANG/WHAMPOA': 'town_KALLANG_WHAMPOA'}, inplace=True)
X = preprocessed_df.drop('monthly_rent', axis=1)
y = preprocessed_df['monthly_rent']
kf = KFold(n_splits=5, shuffle=True, random_state=1)
models = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestRegressor(max_depth=10, random_state=1)
    model.fit(X_train, y_train)
    models.append(model)
joblib.dump(models[-1], 'rental_random_forest_model.joblib')
