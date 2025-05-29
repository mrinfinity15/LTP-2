import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the supply chain dataset
df = pd.read_csv("d:/LTP 2/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1", nrows=5000)
df.columns = df.columns.str.strip().str.lower()  # Standardize column names

# Parse date columns
df['shipping date (dateorders)'] = pd.to_datetime(df['shipping date (dateorders)'], errors='coerce')
df['order date (dateorders)'] = pd.to_datetime(df['order date (dateorders)'], errors='coerce')

# Extract datetime features
df['order_weekday'] = df['order date (dateorders)'].dt.dayofweek
df['order_hour'] = df['order date (dateorders)'].dt.hour
df['shipping_weekday'] = df['shipping date (dateorders)'].dt.dayofweek

# Normalize product name for joining
df['product name'] = df['product name'].astype(str).str.strip().str.lower()

# Load traffic logs
log_df = pd.read_csv("d:/LTP 2/tokenized_access_logs.csv")
log_df['product'] = log_df['Product'].astype(str).str.strip().str.lower()
log_df['hour'] = pd.to_numeric(log_df['Hour'], errors='coerce')

# Aggregate traffic data
traffic = log_df.groupby(['product', 'hour']).size().reset_index(name='product_hour_traffic')

# Merge traffic into main dataset
df = pd.merge(df, traffic, how='left',
              left_on=['product name', 'order_hour'],
              right_on=['product', 'hour'])

df['product_hour_traffic'] = df['product_hour_traffic'].fillna(0)

# Select features
features = [
    'days for shipping (real)', 'days for shipment (scheduled)', 'benefit per order',
    'sales per customer', 'late_delivery_risk', 'shipping mode', 'order city',
    'order state', 'order country', 'product name', 'order_weekday', 'order_hour',
    'shipping_weekday', 'product_hour_traffic'
]

model_df = df[features].copy()
model_df.dropna(subset=['days for shipping (real)'], inplace=True)

# Encode categorical variables
categoricals = ['shipping mode', 'order city', 'order state', 'order country', 'product name']
for col in categoricals:
    model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))

# Prepare X and y
X = model_df.drop('days for shipping (real)', axis=1)
y = model_df['days for shipping (real)']

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search parameters
param_grid = {
    'n_estimators': [50],
    'max_depth': [10, None]
}

# Model training with grid search
print("\U0001F50D Running hyperparameter tuning...")
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=2,
    n_jobs=-1,
    verbose=1,
    scoring='neg_mean_absolute_error'
)

grid_search.fit(X_train, y_train)

# Evaluate best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Tuning Complete.")
print("Best Parameters:", grid_search.best_params_)
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature Importance Plot
importances = best_rf.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (with Traffic)")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feat_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance_with_traffic.png")
plt.show()
