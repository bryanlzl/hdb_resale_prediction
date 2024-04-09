#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

#%%

data = pd.read_csv('newresales_dataset.csv')

regression_columns = ['year', 'town', 'lat', 'lng', 'flat_type', 'storey_range', 'floor_area_sqm', 
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

""" Spatial Data Preprocessing """

# Example parameters - you'll need to adjust these based on your dataset
min_lat, max_lat = 1.26, 1.48  # Latitude bounds of Singapore
min_lng, max_lng = 103.65, 104.00  # Longitude bounds of Singapore
num_rows, num_cols = 1257, 2000  # Grid resolution
# num_rows, num_cols = 126, 200  # Grid resolution

# Calculate the cell size
lat_step = (max_lat - min_lat) / num_rows
lng_step = (max_lng - min_lng) / num_cols

#%%

# Initialize the grid with zeros
# Assuming 4 types of amenities for this example: MRT, schools, shopping malls, and hawker centers
num_channels = 5
grid = np.zeros((num_rows, num_cols, num_channels))

# Function to add an amenity to the grid
def add_amenity_to_grid(grid, data, channel, value=1):
    for index, row in data.iterrows():        
        lat = row['lat']
        lng = row['lng']
        row = int((lat - min_lat) / lat_step)
        col = int((lng - min_lng) / lng_step)
        grid[row, col, channel] += value

grid = np.zeros((num_rows, num_cols, num_channels))

mrt_data = pd.read_csv('mrt_data.csv')
add_amenity_to_grid(grid, mrt_data, channel=1)

schools_data = pd.read_csv('allschool_locations.csv')
add_amenity_to_grid(grid, schools_data, channel=2)

mall_data = pd.read_csv('shopping_mall_coordinates.csv')
add_amenity_to_grid(grid, mall_data, channel=3)

hawkers_data = pd.read_csv('hawkers_dataset.csv')
add_amenity_to_grid(grid, hawkers_data, channel=4)

# plt.imshow(grid)

#%%

def map_hdb_to_grid(grid, hdb_data, prices):
    for index, row in hdb_data.iterrows():
        lat = row['lat']
        lng = row['lng']
        row_idx = int((lat - min_lat) / lat_step)
        col_idx = int((lng - min_lng) / lng_step)
        grid[row_idx, col_idx, 0] += prices.loc[index]

# map_hdb_to_grid(grid, hdb_data)



# # Categorical and numerical columns
categorical_features = ['town', 'flat_type']
numerical_features = ['year', 'storey_range', 'floor_area_sqm', 'remaining_lease', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist']

# Define separate transformers for each categorical column to keep them distinct
categorical_transformers = [(f'cat_{col}', OneHotEncoder(), [col]) for col in categorical_features]

# Combine all transformers into a single preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        *categorical_transformers
    ])


# X = data.drop('mrt_dist', axis=1)
# X = data.drop('shopping_dist', axis=1)
# X = data.drop('school_dist', axis=1)
# X = data.drop('hawker_dist', axis=1)
X = data.drop('resale_price', axis=1)
# X_preprocessed = preprocessor.fit_transform(X)
X_preprocessed = pd.get_dummies(X, columns=categorical_features)
X_preprocessed = X_preprocessed.astype('float32')
y = data['resale_price']
y = y.astype('float32')


# Splitting data while keeping pairs synchronized
X_structured_train, X_structured_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

map_hdb_to_grid(grid, X_structured_train, y_train)

plt.imshow(grid[:,:,0])


#%%

import time
from keras.losses import Huber

# Record the start time
# start_time = time.time()

def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam',  
              loss=Huber(delta=1.0),
              metrics=['mean_squared_error', 'mean_absolute_error'])
    return model

# X_preprocessed = preprocessor.fit_transform(X)

# model = KerasRegressor(build_fn=lambda: build_model(X_preprocessed.shape[1]), epochs=100, batch_size=32, verbose=0)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(model, X_preprocessed, y, cv=kf, scoring='neg_mean_squared_error', error_score='raise')

# end_time = time.time()

# print("MSE scores for each fold:", -scores)
# print("Mean MSE:", -scores.mean())

# training_time = end_time - start_time

# print("Model training time:", training_time, "seconds")

from keras.callbacks import EarlyStopping

# Record the start time
start_time = time.time()

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Generate the indices for the first fold only
train_index, test_index = next(iter(kf.split(X_preprocessed)))

# Split data into training and test for the first fold
X_train_fold, X_test_fold = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index]
y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

# Build and compile your model

model = build_model(X_train_fold.shape[1])
early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=1, mode='min')

import tensorflow as tf

# Train the model on the first fold
with tf.device('/CPU:0'):
    model.fit(X_train_fold, y_train_fold, 
            epochs=10000, batch_size=32, 
            verbose=1, 
            validation_data=(X_test_fold, y_test_fold),
            callbacks=[early_stopping])

# Evaluate the model on the test set of the first fold
score = model.evaluate(X_test_fold, y_test_fold, verbose=0)

# Record the end time 
end_time = time.time()

#%%

print("MSE score for the fold:", score[0], "MAE score for the fold:", score[1])  # Adjust based on your loss function; assuming negative MSE here

training_time = end_time - start_time
print("Model training time for one fold:", training_time, "seconds")

#%%



# # Separate features and target variable
# X = data.drop('resale_price', axis=1)
# y = data['resale_price']

# # Categorical and numerical columns
# categorical_features = ['town', 'flat_type']
# numerical_features = ['year', 'storey_range', 'floor_area_sqm', 'remaining_lease', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist']

# # Preprocessing for numerical data
# numerical_transformer = StandardScaler()

# # Preprocessing for categorical data
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Define the model
# def build_model(input_shape):
#     model = Sequential([
#         Dense(128, activation='relu', input_dim=input_shape),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Preprocessing the data
# X_preprocessed = preprocessor.fit_transform(X)

# # Model creation with dynamic input shape
# model = KerasRegressor(build_fn=lambda: build_model(X_preprocessed.shape[1]), epochs=100, batch_size=10, verbose=0)

# # Evaluate the model using K-Fold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(model, X_preprocessed, y, cv=kf, scoring='neg_mean_squared_error')

# print("Mean Squared Error scores for each fold:", -scores)
# print("Mean MSE:", -scores.mean())


# # data = data.drop(['Unnamed: 0', 'date', 'town', 'block', 'street_name', 'postal', 'nearest_mrt', 
# #                   'nearest_shopping', 'nearest_school', 'nearest_hawker'], axis=1)

# # # Encoding categorical data and normalizing numerical data
# # categorical_features = ['flat_type', 'flat_model']
# # numerical_features = ['year', 'storey_range', 'floor_area_sqm', 'mrt_dist', 'shopping_dist', 'school_dist', 'hawker_dist']

# # # Preprocessor pipeline
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('num', StandardScaler(), numerical_features),
# #         ('cat', OneHotEncoder(), categorical_features)])

# # # Define the model architecture
# # def create_model():
# #     model = Sequential([
# #         Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
# #         Dropout(0.5),
# #         Flatten(),
# #         Dense(50, activation='relu'),
# #         Dense(1, activation='linear')
# #     ])
# #     model.compile(optimizer='adam', loss='mean_squared_error')
# #     return model

# # # Prepare data for modeling
# # X = data.drop(['resale_price'], axis=1)
# # y = data['resale_price'].values
# # X_preprocessed = preprocessor.fit_transform(X)

# # #%%

# # # Since Conv1D expects 3D input, we need to reshape X
# # X_preprocessed = X_preprocessed.reshape((X_preprocessed.shape[0], X_preprocessed.shape[1], 1))

# # # Splitting the data
# # X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# # # Initialize the model with KerasRegressor to use it with scikit-learn functions
# # model = KerasRegressor(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# # # 5-fold cross-validation
# # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# # results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

# # # Metrics
# # mse_scores = -results
# # rmse_scores = np.sqrt(mse_scores)
# # mean_rmse = np.mean(rmse_scores)
# # std_rmse = np.std(rmse_scores)

# # mean_rmse, std_rmse

# # %%
