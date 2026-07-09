import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Sales Forecasting and Business Analytics Dashboard")
st.markdown("""
This dashboard presents an end-to-end business analytics solution developed using:

- Sales Analysis
- Time Series Forecasting
- Prophet Forecasting
- Anomaly Detection
- Product Segmentation
""")
df = pd.read_csv(r"C:\Users\DELL\AppData\Local\Temp\b1b9e01c-a98a-4b82-b43e-ed1eb46c5410_archive.zip.410\train.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Dashboard",
    ["Home","Sales Analysis","Forecast","Anomaly Detection","Product Segmentation"])
if page=="Home":
    st.header("Business Analytics Project")
    st.write("""
This project demonstrates:

• Sales Analysis

• Time Series Forecasting

• Prophet Model

• Anomaly Detection

• Product Clustering

Developed using Python and Streamlit.

""")
total_sales = df["Sales"].sum()

total_orders = df["Order ID"].nunique()

total_customers = df["Customer ID"].nunique()
col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "Total Sales",

    f"${total_sales:,.0f}"

)
col2.metric(

    "Orders",

    total_orders

)
col3.metric(

    "Customers",

    total_customers

)
if page=="Sales Analysis":
    st.header("Sales Analysis")
monthly_sales = (
    df.groupby(pd.Grouper(key="Order Date",freq="ME"))["Sales"].sum().reset_index())
fig = px.line(monthly_sales,x="Order Date",y="Sales",title="Monthly Sales")
st.plotly_chart(fig,use_container_width=True)
category = (
    df.groupby("Category")["Sales"].sum().reset_index())
fig = px.bar(category,x="Category",y="Sales",color="Category")
st.plotly_chart(fig,use_container_width=True)
region = (df.groupby("Region")["Sales"].sum().reset_index())
fig = px.pie(region,names="Region",values="Sales")
st.plotly_chart(fig,use_container_width=True)
if page == "Forecast":
    st.header("📈 Sales Forecasting using Prophet")
st.write("""

This dashboard forecasts future monthly sales using the Prophet forecasting model.

The forecast is generated from historical monthly sales data and includes future predictions along with confidence intervals.

""")
monthly_sales = (df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index())
monthly_sales.columns = ["ds", "y"]
monthly_sales.head()
forecast_months = st.slider(
    "Select Forecast Horizon (Months)",
    min_value=3,max_value=24,value=12)
model = Prophet(
    yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
model.fit(monthly_sales)
future = model.make_future_dataframe(periods=forecast_months,freq="ME")
forecast = model.predict(future)
st.subheader("Forecast Data")
st.dataframe(
    forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(forecast_months))
fig = px.line(forecast,x="ds",y="yhat",title="Forecasted Monthly Sales")
fig.add_scatter(x=monthly_sales["ds"],y=monthly_sales["y"],mode="lines",
                name="Historical Sales")
st.plotly_chart(fig,use_container_width=True)
fig = px.line(forecast,x="ds",y="yhat",title="Forecast with Confidence Interval")
fig.add_scatter(x=forecast["ds"],y=forecast["yhat_upper"],mode="lines",name="Upper Bound")
fig.add_scatter(x=forecast["ds"],y=forecast["yhat_lower"],mode="lines",name="Lower Bound")
st.plotly_chart(fig,use_container_width=True)
st.subheader("Trend Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
latest_actual = monthly_sales["y"].iloc[-1]
forecast_value = forecast["yhat"].iloc[-1]
growth = ((forecast_value - latest_actual)/latest_actual) * 100
c1, c2, c3 = st.columns(3)
c1.metric("Latest Sales",f"${latest_actual:,.0f}")
c2.metric("Forecast Sales",f"${forecast_value:,.0f}")
c3.metric("Expected Growth",f"{growth:.2f}%")
st.subheader("Business Interpretation")
st.write("""
         
The Prophet model forecasts future monthly sales based on historical sales trends and seasonal patterns.

The confidence interval represents the expected range within which future sales are likely to fall.

If the forecast shows increasing demand, organizations should prepare for higher inventory requirements and increased procurement.

Conversely, a declining trend may indicate the need for inventory optimization and revised marketing strategies.

""")
csv = forecast.to_csv(index=False)
st.download_button(
    "Download Forecast CSV",csv,file_name="forecast.csv",mime="text/csv")
if page == "Anomaly Detection":
    st.header("🚨 Sales Anomaly Detection")
monthly_anomaly = (
    df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index())
monthly_anomaly["Z-Score"] = zscore(monthly_anomaly["Sales"])
monthly_anomaly["Status"] = np.where(abs(monthly_anomaly["Z-Score"]) > 3,"Anomaly","Normal")
iso_model = IsolationForest(contamination=0.08,random_state=42)
monthly_anomaly["IF"] = iso_model.fit_predict(monthly_anomaly[["Sales"]])
monthly_anomaly["IF"] = monthly_anomaly["IF"].map({1:"Normal",-1:"Anomaly"})
st.subheader("Monthly Sales")
st.dataframe(monthly_anomaly)
fig = px.scatter(monthly_anomaly,x="Order Date",y="Sales",color="IF",
    title="Isolation Forest Anomaly Detection")
st.plotly_chart(fig,use_container_width=True)
st.metric("Detected Anomalies",len(monthly_anomaly[monthly_anomaly["IF"]=="Anomaly"]))
st.info("""

Anomalies represent unusual sales behaviour.

These may correspond to:

• Festival sales

• Promotions

• Supply shortages

• Data errors

• Unexpected customer demand

""")
if page == "Product Segmentation":
    st.header("🎯 Product Demand Segmentation")
product_summary = (df.groupby("Product Name").agg({"Sales":"sum","Order ID":"nunique"})
.rename(columns={"Order ID":"Orders"}).reset_index())
features = product_summary[["Sales","Orders"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
product_summary["Cluster"] = kmeans.fit_predict(scaled)
pca = PCA(n_components=2)
points = pca.fit_transform(scaled)
plot_df = pd.DataFrame(points,columns=["PC1","PC2"])
plot_df["Cluster"] = product_summary["Cluster"]
fig = px.scatter(plot_df,x="PC1",y="PC2",color=plot_df["Cluster"].astype(str),title="Product Clusters")
st.plotly_chart(fig,use_container_width=True)
cluster_summary = (product_summary.groupby("Cluster")[["Sales","Orders"]].mean())
st.subheader("Cluster Summary")
st.dataframe(cluster_summary)
cluster = st.selectbox("Select Cluster",sorted(product_summary["Cluster"].unique()))
st.dataframe(product_summary[product_summary["Cluster"]==cluster])
csv = product_summary.to_csv(index=False)
st.download_button("Download Cluster Data",csv,file_name="clusters.csv",mime="text/csv")
st.success("""

This dashboard provides:

✅ Sales Analysis

✅ Prophet Forecasting

✅ Anomaly Detection

✅ Product Segmentation

Developed using Streamlit, Prophet, Scikit-learn and Plotly.

""")