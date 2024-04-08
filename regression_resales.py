import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 50)

current_year = 2024
year_cutoff = 2015

# new_resales = pd.read_csv('new_resales.csv')

# new_resales = pd.read_csv('hdb_nearest_mrt_final.csv')
# new_resales = new_resales[new_resales['year'] > year_cutoff]

new_resales = pd.read_csv('newresales_dataset.csv')

print(f'Data has {new_resales.shape[0]} rows', '\n')
# New resale data reduced from 191586 rows to 191256 rows

print('Missing Data')
new_resales.isna().sum()

#%%
''' Checking for hints of multicollinearity '''

regression_columns = ['year', 'town', 'flat_type', 'storey_range', 'floor_area_sqm', 
    'remaining_lease', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist', 'resale_price']

new_resales_reg = new_resales[regression_columns]

sns.heatmap(new_resales_reg.corr(numeric_only=True), vmin=-1, vmax=1, annot=True, 
            fmt=".2f", annot_kws={"size": 10})
sns.pairplot(new_resales_reg)

## Some positive correlation between hawker_dist and remaining_lease strangely. 
## Probably not big enough to cause multicollinearity issues.

## Decent linear relationship between resale_price and multiple variables,
## including year, storey_range, remaining_lease (expected)
## highest of which is floor_area, as expected.


#%%
''' Data Preprocessing '''

original_col = pd.get_dummies(new_resales_reg).columns
new_resales_reg = pd.get_dummies(new_resales_reg, drop_first=True)
# new_resales_reg  = pd.get_dummies(new_resales_reg).drop(columns=['flat_type_2 ROOM', 'town_ANG MO KIO'])  
# Uncomment above to try other dummy variables
dummy_var = set(original_col) - set(new_resales_reg.columns)

# Rename some columns to get smf.ols to read x variables properly
new_resales_reg.columns = new_resales_reg.columns.str.replace(' ','_')
new_resales_reg.rename(columns={'price/sqm':'Price_per_sqm',
                                'town_KALLANG/WHAMPOA':'town_KALLANG_WHAMPOA',
                                'flat_type_MULTI-GENERATION':'flat_type_MULTI_GENERATION'},
                       inplace=True)

# Subtract year by 2024 to simplify analysis of coefficients
new_resales_reg['year'] = new_resales_reg['year'] - current_year


#%%
''' Regression: Simple'''

x_variables = list(new_resales_reg.columns)
y_variable = 'resale_price'

try: x_variables.remove(y_variable)
except ValueError: pass

# Exporting the correlation matrix, too large to visualize
new_resales_reg[x_variables].corr().to_csv('correl.csv')

linear_model = smf.ols(data=new_resales_reg, formula=f'{y_variable} ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Adj R2 0.862, 3-room and 4-room flats non-significant at 1% level

#%%
''' Regression: Changing y variable to log '''

linear_model = smf.ols(data=new_resales_reg, formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Adj R2 0.888, all variables significant. 
# Since R2 increased significantly, we keep log(y)

#%%
''' Regression: Adding quadratic terms to numerical variables '''

var = ['storey_range', 'remaining_lease', 'floor_area_sqm']
string = ''.join(map(lambda x: ' + I(' + x + '**2)', var))

linear_model = smf.ols(data=new_resales_reg, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}' + string).fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Adj R2 0.889, not much difference.
# Remaining_lease ** 2 is not significant, the others are.
# We prefer model simplicity hence we drop the quadratic terms.

#%%
''' Regression: Adding interaction variables between flat_type and floor_area '''

var = ['3_ROOM', '4_ROOM', '5_ROOM', 'EXECUTIVE', 'MULTI_GENERATION']
interaction_var = 'floor_area_sqm'
# interaction_var = 'remaining_lease'

string = ''.join(map(lambda x: f' + {interaction_var} : flat_type_' + x, var))

linear_model = smf.ols(data=new_resales_reg, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}' + string).fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Adj R2 0.891, not much difference.
# Interaction with 4 room and MultiGen are not significant (1% level)

# However, interaction with 3 room is highly significant. The intercept of 3 room also changes to negative.
# This is noteworthy since we know that 3 room floor areas have a wide range with many positive outliers.
# Hence we should keep the interaction variable with 3 room.

# I also tried interaction variables [year, storey_range, remaining_lease], results are not significant.


#%%
''' Regression: Trying a different date range. In particular, we try the Covid years (2020 onwards)'''

resales_after_2020 = new_resales_reg[new_resales_reg['year']>=(2020-current_year)].reset_index(drop=True)

linear_model = smf.ols(data=resales_after_2020, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)} + flat_type_3_ROOM:floor_area_sqm').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# R2 improved to 0.920, showing that the smaller dataset works better to explain variance. 
# We keep this as final model.


#%%
''' Residual Analysis for Heteroscedasticity '''
pred_y = linear_model.fittedvalues
plt.scatter(pred_y, linear_model.resid, s=1)
plt.plot([min(pred_y),max(pred_y)],[0,0], color='black')
plt.title('Residuals vs Pred y')
plt.show()

for var in ['year', 'storey_range', 'remaining_lease', 'floor_area_sqm', 'mrt_dist']:
    plt.scatter(resales_after_2020[var], linear_model.resid, s=1)
    plt.plot([min(resales_after_2020[var]),max(resales_after_2020[var])],[0,0], color='black')
    plt.title(f'Residuals vs {var}')
    plt.show()

# In general variance of residuals seem constant for most variables 
# Only residuals vs storey_range shows a decreasing variance.
# We tried including storey_range**2 and interaction variables with storey_range, did not change the residual plot.


#%%
''' Train-test split over time series (Train 4 year models, test 5th year) '''

from sklearn.linear_model import LinearRegression

year_range = 4

new_resales_reg['flat_type_3_ROOM:floor_area_sqm'] = new_resales_reg['flat_type_3_ROOM'] * new_resales_reg['floor_area_sqm']

for start_year in range(2015, current_year - year_range + 1):
    train_df = new_resales_reg[(new_resales_reg['year'] >= start_year-current_year) & (new_resales_reg['year'] < start_year-current_year+year_range)]
    test_df = new_resales_reg[new_resales_reg['year'] == start_year-current_year+year_range]
    
    X_train = train_df.drop(columns=['resale_price']).to_numpy()
    y_train = np.log(train_df['resale_price'].to_numpy())
    X_test = test_df.drop(columns=['resale_price']).to_numpy()
    y_test = np.log(test_df['resale_price'].to_numpy())
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test) 
    errors = y_pred - y_test
    
    plt.scatter(y_pred, y_test, s=1)
    plt.plot([min(y_pred),max(y_pred)], [min(y_pred),max(y_pred)], color='black')
    plt.title(f'Model: {start_year}-{start_year+year_range-1}, Test {start_year+year_range}')
    plt.ylabel('Actual log(resale_price)')
    plt.xlabel('Predicted log(resale_price)')
    plt.show()
    
    print(f'Model: {start_year}-{start_year+year_range-1} ({train_df.shape[0]} datapoints), Test {start_year+year_range} ({test_df.shape[0]} datapoints)')
    print(f'R2: {lm.score(X_test, y_test):.3f}')
    print(f'Mean Error: {errors.mean():.3f}')
    print(f'MAE: {np.abs(errors).mean():.3f}')
    print(f'RMSE: {np.sqrt(np.square(errors).mean()):.3f}')
    print()
    
# The 4-year models perform quite well on test data, testing R2 from 0.77-0.91
# Mean error usually below 4%, MAE 8%, RMSE 11%

# Exception in 2021-2022 where the model underpredicted by about 10% on average,
# suggesting an abnormal surge in house prices during Covid.
# Also exception in 2024 where the model overpredicted by 7%, could be due to cooling measures in 2023
    
#%%
''' Train-test split, cross-sectionally, for 2020-2023'''
from sklearn.model_selection import KFold

n_splits = 5  # For 5-fold split
kf = KFold(n_splits=n_splits, shuffle=True, random_state=6969)

# Prepare data
X = train_df.drop(columns=['resale_price']).to_numpy()
y = np.log(train_df['resale_price'].to_numpy())  # Log-transform y for training

fold = 0
for train_index, test_index in kf.split(X):
    fold += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train model
    lm.fit(X_train, y_train)
    
    # Predict and calculate errors
    y_pred = lm.predict(X_test)
    errors = y_pred - y_test
    
    print(f'Fold {fold}, {X_train.shape[0]} train datapoints, Test: {X_test.shape[0]} test datapoints')
    print(f'R2: {lm.score(X_test, y_test):.3f}')
    print(f'Mean Error: {errors.mean():.3f}')
    print(f'MAE: {np.abs(errors).mean():.3f}')
    print(f'RMSE: {np.sqrt(np.square(errors).mean()):.3f}')
    print()

# Results seem too good to be true?! lol
# R2 always above 0.92, mean error basically 0%, MAE 7%, RMSE 9%