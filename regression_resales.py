import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

current_year = 2024

new_resales = pd.read_csv('new_resales.csv')

regression_columns = ['year', 'flat_type', 'storey_range', 'town',
                      'remaining_lease', 'floor_area_sqm', 'resale_price']

#%%
''' Checking for hints of multicollinearity '''

new_resales_reg = new_resales[regression_columns]

sns.heatmap(new_resales_reg.corr(numeric_only=True), vmin=-1, vmax=1, annot=True)
sns.pairplot(new_resales_reg)

## There is slight positive correlation between storey range and remaining lease actually.
## Probably due to newer flats having higher floors. 
## Probably not big enough to cause multicollinearity issues.


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

# Adj R2 0.86, Non-Significant Variables: 2 Room

#%%
''' Regression: Changing y variable to log '''

linear_model = smf.ols(data=new_resales_reg, formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Adj R2 0.897, all variables significant. 
# Since R2 increased significantly, we keep log(y)

#%%
''' Regression: Adding quadratic terms to numerical variables '''

var = ['storey_range', 'remaining_lease', 'floor_area_sqm']
string = ''.join(map(lambda x: ' + I(' + x + '**2)', var))

linear_model = smf.ols(data=new_resales_reg, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}' + string).fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Adj R2 0.898, storey_range ** 2 not significant
# Because there is no significant increase in R2, we prefer model simplicity hence we drop the quadratic terms

#%%
''' Regression: Adding interaction variables between flat_type and floor_area '''

var = ['2_ROOM', '3_ROOM', '4_ROOM', '5_ROOM', 'EXECUTIVE', 'MULTI_GENERATION']
string = ''.join(map(lambda x: ' + floor_area_sqm : flat_type_' + x, var))

linear_model = smf.ols(data=new_resales_reg, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# Interaction variables are significant. However, Adj R2 is 0.90, not a big increase.
# Probably choose the simpler model.

# It is interesting to note the sizeable changes in coefficients of flat type 
# and the slope against floor area when interaction variables are included.

# I also tried interaction variables between flat_type and [year, storey_range, remaining_lease], results are not significant.


#%%
''' Regression: Trying a different date range. In particular, we try the covid years (2020 onwards)'''

resales_after_2020 = new_resales_reg[new_resales_reg['year']>=(2020-current_year)].reset_index(drop=True)

linear_model = smf.ols(data=resales_after_2020, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# R2 improved to 0.896, showing that the smaller dataset works better to explain variance. We keep this as final model.


#%%
''' Residual Analysis for Heteroscedasticity '''
pred_y = linear_model.fittedvalues
plt.scatter(pred_y, linear_model.resid, s=1)
plt.plot([min(pred_y),max(pred_y)],[0,0], color='black')
plt.title('Residuals vs Pred y')
plt.show()

for var in ['year', 'storey_range', 'remaining_lease', 'floor_area_sqm']:
    plt.scatter(resales_after_2020[var], linear_model.resid, s=1)
    plt.plot([min(resales_after_2020[var]),max(resales_after_2020[var])],[0,0], color='black')
    plt.title(f'Residuals vs {var}')
    plt.show()

# In general no evidence of heteroscedasticity vs most variables, except perhaps storey_range.
# However, a model which includes storey_range**2 shows up with an insignificant coefficient.
# Should not affect our model.


#%%
''' Train-test split by constructing rolling 4 year models to predict the 5th year '''

from sklearn.linear_model import LinearRegression

year_range = 4

for start_year in range(2015, current_year - year_range):
    train_df = new_resales_reg[(new_resales_reg['year'] >= start_year-current_year) & (new_resales_reg['year'] < start_year-current_year+year_range)]
    test_df = new_resales_reg[new_resales_reg['year'] == start_year-current_year+year_range]
    
    X_train = train_df.drop(columns=['resale_price']).to_numpy()
    y_train = train_df['resale_price'].to_numpy()
    X_test = test_df.drop(columns=['resale_price']).to_numpy()
    y_test = test_df['resale_price'].to_numpy()
    
    lm = LinearRegression()
    lm.fit(X_train, np.log(y_train))
    errors = lm.predict(X_test) - np.log(y_test)
    
    print(f'Model: {start_year}-{start_year+year_range-1} ({train_df.shape[0]} datapoints), Test {start_year+year_range} ({test_df.shape[0]} datapoints)')
    print(f'R2: {lm.score(X_train, np.log(y_train)):.3f}')
    print(f'Mean Error: {errors.mean():.3f}')
    print(f'SD Error: {errors.std():.3f}')
    print()
    
    # The 4-year models perform quite well on test data with mean error usually below 4%. 
    # SD of the errors are also about 10%.
    
    # Exception in 2021-2022 where the model underpredicted by about 10% on average,
    # suggesting an abnormal surge in house prices during Covid.
    
    