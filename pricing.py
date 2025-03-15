import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv('home-data-for-ml-course/train.csv')
test_df = pd.read_csv('home-data-for-ml-course/test.csv')

# Prepare features and target
X_train = train_df.drop(columns=['Id', 'SalePrice'])
y_train = train_df['SalePrice']
X_test = test_df.drop(columns=['Id'])
test_ids = test_df['Id']

# Replace 'NA' with NaN
X_train = X_train.replace('NA', np.nan)
X_test = X_test.replace('NA', np.nan)

# Convert categorical columns
categorical_cols = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
    'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 
    'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# Initial validation to tune the model
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_split, y_train_split)
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f'Validation RMSE: {rmse}')

# Train on full data for final submission
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generate submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred
})
submission.to_csv('submission.csv', index=False)