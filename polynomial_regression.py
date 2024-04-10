#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%%

data = pd.read_csv('newresales_dataset.csv')

regression_columns = ['year', 'town', 'flat_type', 'storey_range', 'floor_area_sqm', 
    'remaining_lease', 'resale_price', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist']

# Filter the dataset for the columns of interest
data = data[regression_columns]

# sns.heatmap(data.corr(numeric_only=True), vmin=-1, vmax=1, annot=True, 
#             fmt=".2f", annot_kws={"size": 10})
# sns.pairplot(data)

#%%

data.columns = data.columns.str.replace(' ','_')
data.columns = data.columns.str.replace('-','_')
data.rename(columns={'price/sqm':'Price_per_sqm',
                    'town_KALLANG/WHAMPOA':'town_KALLANG_WHAMPOA'},
                    inplace=True)

#%%

categorical_features = ['town', 'flat_type']
numerical_features = ['year', 'storey_range', 'floor_area_sqm', 'remaining_lease', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist']

X = data.drop('resale_price', axis=1)
X_preprocessed = pd.get_dummies(X, columns=categorical_features)
X_preprocessed = X_preprocessed.astype('float32')
y = data['resale_price']
y = y.astype('float32')


degree = 2  # You can adjust this to fit the degree of polynomial you want to consider

polynomial_features = PolynomialFeatures(degree=degree)
X_poly = polynomial_features.fit_transform(X_preprocessed)

# Splitting data while keeping pairs synchronized
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, np.log(y_train))

#%%

y_pred = model.predict(X_test)

mse = mean_squared_error(np.log(y_test), y_pred)
r2 = r2_score(np.log(y_test), y_pred)

errors = y_pred - np.log(y_test)

print(f"RMSE: {np.sqrt(np.square(errors).mean()):.3f}")
print(f'Mean Error: {errors.mean():.3f}')
print(f'MAE: {np.abs(errors).mean():.3f}')
print(f"R-squared: {r2}")

#%%

plt.scatter(np.log(y_test), y_pred, s=1)
plt.title(f'Model: {2016}-{2024}')
plt.xlabel('Actual log(resale_price)')
plt.ylabel('Predicted log(resale_price)')
plt.plot(np.log(y_test), np.log(y_test), color='red')
plt.show()

# %%


categorical_features = ['town', 'flat_type']
numerical_features = ['year', 'storey_range', 'floor_area_sqm', 'remaining_lease', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist']

X = data.drop('resale_price', axis=1)
X = X[X['year']>=(2020-current_year)].reset_index(drop=True)
X_preprocessed = pd.get_dummies(X, columns=categorical_features)
X_preprocessed = X_preprocessed.astype('float32')
y = data['resale_price']
y = y.astype('float32')


degree = 2  # You can adjust this to fit the degree of polynomial you want to consider

polynomial_features = PolynomialFeatures(degree=degree)
X_poly = polynomial_features.fit_transform(X_preprocessed)

# Splitting data while keeping pairs synchronized
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, np.log(y_train))

#%%

y_pred = model.predict(X_test)

mse = mean_squared_error(np.log(y_test), y_pred)
r2 = r2_score(np.log(y_test), y_pred)

errors = y_pred - np.log(y_test)

print(f"RMSE: {np.sqrt(np.square(errors).mean()):.3f}")
print(f'Mean Error: {errors.mean():.3f}')
print(f'MAE: {np.abs(errors).mean():.3f}')
print(f"R-squared: {r2}")

#%%

plt.scatter(np.log(y_test), y_pred, s=1)
plt.title(f'Model: {2016}-{2024}')
plt.xlabel('Actual log(resale_price)')
plt.ylabel('Predicted log(resale_price)')
plt.plot(np.log(y_test), np.log(y_test), color='red')
plt.show()

# %%
