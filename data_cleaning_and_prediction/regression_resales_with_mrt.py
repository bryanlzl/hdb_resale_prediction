#%%
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

current_year = 2024

new_resales = pd.read_csv('hdb_nearest_mrt_final.csv')

regression_columns = ['year', 'flat_type', 'storey_range', 'town',
                      'remaining_lease', 'floor_area_sqm', 'resale_price', 'mrt_distance', 'nearest_mrt']

#%%
''' Checking for hints of multicollinearity '''

new_resales = new_resales.drop(['latitude', 'longitude', 'sale_year', 'sale_month', 'mrt_lat', 'mrt_lng'], axis=1)

new_resales_reg = new_resales[regression_columns]

sns.heatmap(new_resales_reg.corr(numeric_only=True), vmin=-1, vmax=1, annot=True)
sns.pairplot(new_resales_reg)

## No strong collearity between mrt_distance and the other variables


#%%
''' Data Preprocessing '''

original_col = pd.get_dummies(new_resales_reg).columns
new_resales_reg = pd.get_dummies(new_resales_reg, drop_first=True)
dummy_var = set(original_col) - set(new_resales_reg.columns)

# Rename some columns to get smf.ols to read x variables properly
new_resales_reg.columns = new_resales_reg.columns.str.replace(' ','_')
new_resales_reg.columns = new_resales_reg.columns.str.replace('-','_')
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

# Adj R2 0.843, not all variables significant, but to note that not including mrt stations, 
# Adj R2 0.823 and all vars significant. so including the stations did better but yeah, a lot
# more insignificant variables to list

#%%
''' Regression: Changing y variable to log '''

linear_model = smf.ols(data=new_resales_reg, formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# With just mrt_distance:
# Adj R2 0.792, Serangoon no longer significant, likely bc serangoon's area is too large. 
# Since R2 decreased significantly, we skip log(y)

# With mrt_distance and nearest_mrt
# Adj R2 0.804, still a few insignificant variables
# Since R2 decreased significantly, we skip log(y)

#%%
''' Regression: Adding quadratic terms to numerical variables '''

var = ['storey_range', 'remaining_lease', 'floor_area_sqm']
string = ''.join(map(lambda x: ' + I(' + x + '**2)', var))

linear_model = smf.ols(data=new_resales_reg, 
    formula=f'{y_variable} ~ {"+".join(x_variables)}' + string).fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# With just mrt_distance:
# Adj R2 0.827, All variables significant
# Because there is no significant increase in R2, we prefer model simplicity hence we drop the quadratic terms

# With mrt_distance and nearest_mrt
# Adj R2 0.845, still have some insignificant variables
# Because there is still no significant increase in R2, we prefer model simplicity hence we drop the quadratic terms


#%%
''' Regression: Adding interaction variables between flat_type and floor_area '''

var = ['2_ROOM', '3_ROOM', '4_ROOM', '5_ROOM', 'EXECUTIVE', 'MULTI_GENERATION']
string = ''.join(map(lambda x: ' + floor_area_sqm : flat_type_' + x, var))

linear_model = smf.ols(data=new_resales_reg, 
    formula=f'{y_variable} ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# With just mrt_distance:
# Interaction variables are significant. However, Adj R2 is 0.823, no change from simple model.
# Probably choose the simpler model.

# With mrt_distance and nearest_mrt
# Interaction variables are significant. However, Adj R2 is 0.843, no change from simple model.
# Probably choose the simpler model.

#%%
''' Regression: Trying a different date range. In particular, we try the covid years (2020 onwards)'''

resales_after_2020 = new_resales_reg[new_resales_reg['year']>=(2020-current_year)].reset_index(drop=True)

linear_model = smf.ols(data=resales_after_2020, 
    formula=f'{y_variable} ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# With just mrt_distance:
# R2 improved to 0.878, showing that the smaller dataset works better to explain variance. We keep this as final model.

# With mrt_distance and nearest_mrt
# R2 improved to 0.905, showing that the smaller dataset works better to explain variance. We keep this as final model.

#%%
''' Residual Analysis for Heteroscedasticity '''
pred_y = linear_model.fittedvalues
plt.scatter(pred_y, linear_model.resid, s=1)
plt.plot([min(pred_y),max(pred_y)],[0,0], color='black')
plt.title('Residuals vs Pred y')
plt.show()

for var in ['year', 'storey_range', 'remaining_lease', 'floor_area_sqm', 'mrt_distance']:
    plt.scatter(resales_after_2020[var], linear_model.resid, s=1)
    plt.plot([min(resales_after_2020[var]),max(resales_after_2020[var])],[0,0], color='black')
    plt.title(f'Residuals vs {var}')
    plt.show()

# In general no evidence of heteroscedasticity vs most variables, except perhaps storey_range.
# However, a model which includes storey_range**2 shows up with an insignificant coefficient.
# Should not affect our model. mrt_distance no evidence as well


#%%
''' Train-test split by constructing rolling 4 year models to predict the 4th year '''

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
    lm.fit(X_train, y_train)
    errors = lm.predict(X_test) - y_test
    
    print(f'Model: {start_year}-{start_year+year_range} ({train_df.shape[0]} datapoints), Test {start_year+year_range} ({test_df.shape[0]} datapoints)')
    print(f'R2: {lm.score(X_train, y_train):.3f}')
    print(f'Mean Error: {errors.mean():.3f}')
    print(f'SD Error: {errors.std():.3f}')
    print()
    
    # With just mrt_distance:
    # Adding MRT distance adds a significant amount of error (no joke A LOT), model probably can't 
    # predict changes in new MRT locations

    # With mrt_distance and nearest_mrt
    # Adding MRT distance adds a significant amount of error (no joke A LOT, i believe even more than without nearest_mrt), model probably can't 
    # predict changes in new MRT locations
    
#%%
''' Ashe + Bryan Testing, No point doing this but we'll just try anyways'''


'''

# DISCLAIMER: JUST COPIED THE EARLIER CODES FOR THE MODELS TO WORK EXACTLY THE SAME WITH THE NEW COLUMN
# IT DOESNT GET SAVED SO PLS CHECK

'''

''' Train-test split by constructing rolling 4 year models to predict the 4th year (but with 5-fold testing) '''

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

year_range = 4
n_splits = 5  # For 5-fold split

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=6969)

for start_year in range(2015, current_year - year_range):
    train_df = new_resales_reg[(new_resales_reg['year'] >= start_year-current_year) & (new_resales_reg['year'] < start_year-current_year+year_range)]
    test_df = new_resales_reg[new_resales_reg['year'] == start_year-current_year+year_range]
    
    # Prepare data
    X = train_df.drop(columns=['resale_price']).to_numpy()
    y = train_df['resale_price'].to_numpy()  # Log-transform y for training
    
    # Initialize model
    lm = LinearRegression()
    
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
        
        print(f'Fold {fold}, Model: {start_year}-{start_year+year_range} ({X_train.shape[0]} train datapoints), Test: {start_year+year_range} ({X_test.shape[0]} test datapoints)')
        print(f'R2: {lm.score(X_train, y_train):.3f}')
        print(f'Mean Error: {errors.mean():.3f}')
        print(f'SD Error: {errors.std():.3f}')
        print()

# Overall even with 5 fold testing, SD error rarely fell below 45373.468...
# I think mrt data just doesn't work too well with prediction

#%%
''' Adding inflation rates '''

Inflation_rates = {
    -9: -0.522618167075298,
    -8: -0.532268739716253,
    -7: 0.576260310166349,
    -6: 0.43862011844684,
    -5: 0.565260568780326,
    -4: -0.181916666666634,
    -3: 2.30485959040484,
    -2: 6.12106004039418,
    -1: 4.825,
    0: 3.1 # Forecasted
}

new_resales_reg['Inflation_rate'] = new_resales_reg['year'].map(Inflation_rates)

# %%
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

# Here you can see that there's a v high colllinearity between inflation and year with 0.82. that's because of the trends in 
# the inflation of recent years. I attached a screenshot of the inflation trends so if we actually include old data 
# it might work but I don't think it's worth it

# No variables insignificant when mrt_distance is included

# R2 0.887 with nearest_mrt, few insignificant variables, comments still the same

#%%
''' Regression: Changing y variable to log '''

linear_model = smf.ols(data=new_resales_reg, formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# With just mrt_distance:
# Adj R2 0.878, all variables significant. Lower R2 than before including inflation rate

# With mrt_distance and nearest_mrt
# Adj R2 0.910, somehow works better with inflation rate? (R2 0.905 for 2020 onwards from prev model)

#%%

''' Regression: Trying a different date range. In particular, we try the covid years (2020 onwards w/ inflation rate)'''

resales_after_2020 = new_resales_reg[new_resales_reg['year']>=(2020-current_year)].reset_index(drop=True)

linear_model = smf.ols(data=resales_after_2020, 
    formula=f'np.log({y_variable}) ~ {"+".join(x_variables)}').fit()
print(linear_model.summary())
print(f'Dummy Variables: {dummy_var}')

# With just mrt_distance:
# R2 improved to 0.898, very minimal change from before (0.876)

# With mrt_distance and nearest_mrt:
# R2 improved to 0.932, this is our current best model R2 using any form of mrt data

#%%

''' Train-test split by constructing rolling 4 year models to predict the 5th year '''

# Added a test to predict 2024 sales as well

year_range = 4

for start_year in range(2015, current_year - year_range):
    train_df = new_resales_reg[(new_resales_reg['year'] >= start_year-current_year) & (new_resales_reg['year'] < start_year-current_year+year_range)]
    test_df = new_resales_reg[new_resales_reg['year'] == start_year-current_year+year_range]
    
    X_train = train_df.drop(columns=['resale_price']).to_numpy()
    y_train = train_df['resale_price'].to_numpy()
    X_test = test_df.drop(columns=['resale_price']).to_numpy()
    y_test = test_df['resale_price'].to_numpy()
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    errors = lm.predict(X_test) - y_test
    
    print(f'Model: {start_year}-{start_year+year_range-1} ({train_df.shape[0]} datapoints), Test {start_year+year_range} ({test_df.shape[0]} datapoints)')
    print(f'R2: {lm.score(X_train, np.log(y_train)):.3f}')
    print(f'Mean Error: {errors.mean():.3f}')
    print(f'SD Error: {errors.std():.3f}')
    print()
    
    # The 4-year models here have higher R2 but overall produces higher errors
    # SD of the errors are quite crazy, but still about 50k
    
# %%
