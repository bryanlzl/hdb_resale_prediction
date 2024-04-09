import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import joblib

pd.set_option('display.max_rows', 50)

current_year = 2024
year_cutoff = 2015

new_resales = pd.read_csv('newresales_dataset.csv')

regression_columns = ['year', 'town', 'flat_type', 'storey_range', 'floor_area_sqm', 
    'remaining_lease', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist', 'resale_price']

new_resales_reg = new_resales[regression_columns]

original_col = pd.get_dummies(new_resales_reg).columns
new_resales_reg = pd.get_dummies(new_resales_reg, drop_first=True)

dummy_var = set(original_col) - set(new_resales_reg.columns)

new_resales_reg.columns = new_resales_reg.columns.str.replace(' ','_')
new_resales_reg.rename(columns={'price/sqm':'Price_per_sqm',
                                'town_KALLANG/WHAMPOA':'town_KALLANG_WHAMPOA',
                                'flat_type_MULTI-GENERATION':'flat_type_MULTI_GENERATION'},
                       inplace=True)
new_resales_reg['year'] = new_resales_reg['year'] - current_year

x_variables = list(new_resales_reg.columns)
y_variable = 'resale_price'

try: x_variables.remove(y_variable)
except ValueError: pass

resales_after_2020 = new_resales_reg[new_resales_reg['year']>=(2020-current_year)].reset_index(drop=True)

linear_model = smf.ols(data=resales_after_2020, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)} + flat_type_3_ROOM:floor_area_sqm').fit()

joblib.dump(linear_model, 'linear_model.joblib')