# Preprocessing of Numerical and Categorical Data
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn import compose
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

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
rentals_dataset = pd.read_csv("rentals_dataset.csv", index_col=0)
rentals_dataset["date"] = pd.to_datetime(rentals_dataset["date"])
reference_date = datetime(2023, 12, 31)
rentals_dataset["months_duration"] = (
    rentals_dataset["date"].dt.year - reference_date.year
) * 12 + (rentals_dataset["date"].dt.month - reference_date.month)
reduced_df = rentals_dataset[
    [
        "town",
        "flat_type",
        "property_age",
        "avg_floor_area_sqm",
        "mrt_dist",
        "shopping_dist",
        "intschool_dist",
        "hawker_dist",
        "months_duration",
        "monthly_rent",
    ]
]
original_col = pd.get_dummies(reduced_df).columns
preprocessed_df = pd.get_dummies(reduced_df, drop_first=True, dtype=int)
dummy_var = set(original_col) - set(preprocessed_df.columns)
preprocessed_df.columns = preprocessed_df.columns.str.replace(" ", "_")
preprocessed_df.rename(
    columns={"town_KALLANG/WHAMPOA": "town_KALLANG_WHAMPOA"}, inplace=True
)
x_variables = list(preprocessed_df.columns)
y_variable = "monthly_rent"
try:
    x_variables.remove(y_variable)
except ValueError:
    pass
rental_linear_model = smf.ols(
    data=preprocessed_df, formula=f'{y_variable} ~ {"+".join(x_variables)}'
).fit()
joblib.dump(rental_linear_model, "rental_linear_model.joblib")

print(x_variables)
print(y_variable)
