import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Lead Time Predictor", layout="wide")
st.title("\U0001F4E6 Lead Time Predictor Dashboard")

st.markdown("""
### \u2139\ufe0f Dataset Summary
This dashboard predicts the real lead time (delivery duration) for orders based on:
- Order timing
- Product type
- Delivery location
- Web traffic logs (as a proxy for system load or demand)

The model helps logistics teams plan better by showing which factors delay shipping the most.
""")

uploaded_file = st.file_uploader("Upload Supply Chain CSV", type=["csv"])
traffic_file = st.file_uploader("Upload Tokenized Access Log CSV", type=["csv"])

if uploaded_file and traffic_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", nrows=5000)
    traffic_df = pd.read_csv(traffic_file)

    df.columns = df.columns.str.strip().str.lower()
    traffic_df.columns = traffic_df.columns.str.strip().str.lower()

    df['shipping date (dateorders)'] = pd.to_datetime(df['shipping date (dateorders)'], errors='coerce')
    df['order date (dateorders)'] = pd.to_datetime(df['order date (dateorders)'], errors='coerce')
    df['order_weekday'] = df['order date (dateorders)'].dt.dayofweek
    df['order_hour'] = df['order date (dateorders)'].dt.hour
    df['shipping_weekday'] = df['shipping date (dateorders)'].dt.dayofweek
    df['product name'] = df['product name'].astype(str).str.strip().str.lower()

    traffic_df['product'] = traffic_df['product'].astype(str).str.strip().str.lower()
    traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')
    traffic = traffic_df.groupby(['product', 'hour']).size().reset_index(name='product_hour_traffic')
    df = pd.merge(df, traffic, how='left',
                  left_on=['product name', 'order_hour'],
                  right_on=['product', 'hour'])
    df['product_hour_traffic'] = df['product_hour_traffic'].fillna(0)

    features = [
        'days for shipping (real)', 'days for shipment (scheduled)', 'benefit per order',
        'sales per customer', 'late_delivery_risk', 'shipping mode', 'order city',
        'order state', 'order country', 'product name', 'order_weekday', 'order_hour',
        'shipping_weekday', 'product_hour_traffic'
    ]

    model_df = df[features].copy()
    model_df.dropna(subset=['days for shipping (real)'], inplace=True)

    categoricals = ['shipping mode', 'order city', 'order state', 'order country', 'product name']
    for col in categoricals:
        model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))

    X = model_df.drop('days for shipping (real)', axis=1)
    y = model_df['days for shipping (real)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LightGBM"])

    st.markdown(f"""
    ### \U0001F9E0 About {model_choice}
    - **Random Forest**: Combines many decision trees for stable predictions.
    - **XGBoost**: A boosting model that learns from mistakes and often performs very well.
    - **LightGBM**: Like XGBoost but faster, great for large datasets.
    """)

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    elif model_choice == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=-1, learning_rate=0.1, random_state=42)

    st.subheader("\U0001F501 Training Model...")
    with st.spinner("Training in progress..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success("\u2705 Model Training Complete")
    st.metric("\U0001F4C9 MAE (hrs)", f"{mae:.2f}")
    st.metric("\U0001F4C8 R² Score", f"{r2:.2f}")

    st.markdown(f"""
    **MAE** of **{mae:.2f} hours** = about **{int(mae*60)} minutes** off per prediction.
    **R² Score** of **{r2:.2f}** = explains **{r2*100:.0f}%** of delivery time variability.
    """)

    st.subheader("\U0001F4CA Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [X.columns[i] for i in indices[:3]]

    fig1, ax1 = plt.subplots()
    ax1.bar(range(X.shape[1]), importances[indices])
    ax1.set_xticks(range(X.shape[1]))
    ax1.set_xticklabels([X.columns[i] for i in indices], rotation=90)
    ax1.set_title(f"Feature Importance ({model_choice})")
    st.pyplot(fig1)

    st.markdown(f"""
    #### \U0001F4A1 Key Influencers
    These are the **top 3 features** most affecting delivery time:
    1. **{top_features[0]}**
    2. **{top_features[1]}**
    3. **{top_features[2]}**
    """)

    with st.expander("\U0001F4C1 See Residual Error Distribution"):
        st.markdown("Residuals are prediction errors. Ideal models have residuals centered around 0.")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax2, color="orange")
        ax2.set_title("Residual Distribution")
        st.pyplot(fig2)

    st.subheader("\U0001F4CB Prediction Table")
    st.markdown("This table shows a comparison between actual delivery times and predicted ones, along with key input features.")
    X_test_display = X_test.copy()
    X_test_display["Actual Lead Time"] = y_test.values
    X_test_display["Predicted Lead Time"] = y_pred
    st.dataframe(X_test_display.reset_index(drop=True))

    # Download button
    csv = X_test_display.reset_index(drop=True).to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions as CSV", csv, file_name="lead_time_predictions.csv", mime="text/csv")

else:
    st.info("\U0001F4C2 Please upload both required CSV files to begin.")
