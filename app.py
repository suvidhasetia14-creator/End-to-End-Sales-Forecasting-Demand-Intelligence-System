import streamlit as st
import pandas as pd
import numpy as np
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

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\DELL\AppData\Local\Temp\b1b9e01c-a98a-4b82-b43e-ed1eb46c5410_archive.zip.410\train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

df = load_data()

# ----------------------------------------------------
# Dashboard Title
# ----------------------------------------------------
st.title("📊 Sales Forecasting and Business Analytics Dashboard")

st.markdown("""
This dashboard presents an end-to-end business analytics solution.

### Features
- 📈 Sales Analysis
- 🔮 Time Series Forecasting (Prophet)
- 🚨 Anomaly Detection
- 🎯 Product Segmentation
- 📊 Business Intelligence
""")

# ----------------------------------------------------
# Sidebar Navigation
# ----------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Sales Analysis",
        "Forecast",
        "Anomaly Detection",
        "Product Segmentation"
    ]
)

# ====================================================
# HOME PAGE
# ====================================================
if page == "Home":

    st.header("Business Analytics Project")

    st.markdown("""
## 🔄 Project Workflow

Dataset

⬇️

Data Cleaning

⬇️

Exploratory Data Analysis

⬇️

Time Series Forecasting

⬇️

Anomaly Detection

⬇️

Product Segmentation

⬇️

Business Recommendations
""")

    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "📦 Orders",
        total_orders
    )

    c3.metric(
        "👥 Customers",
        total_customers
    )

    st.markdown("---")

    st.success("""
This dashboard demonstrates:

✅ Sales Analysis

✅ Forecasting using Prophet

✅ Anomaly Detection

✅ Product Segmentation

Developed using Python, Streamlit, Prophet,
Scikit-learn and Plotly.
""")
# ====================================================
# SALES ANALYSIS PAGE
# ====================================================
elif page == "Sales Analysis":

    st.header("📊 Sales Analysis")

    # ---------------------------------------
    # Monthly Sales Trend
    # ---------------------------------------
    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Order Date",
        yaxis_title="Sales ($)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Sales by Category
    # ---------------------------------------
    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        color="Category",
        title="Sales by Category",
        text_auto=".2s"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Sales by Region
    # ---------------------------------------
    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        title="Regional Sales Distribution",
        hole=0.45
    )

    fig.update_traces(textposition="inside")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Sales by Segment
    # ---------------------------------------
    segment_sales = (
        df.groupby("Segment")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        segment_sales,
        x="Segment",
        y="Sales",
        color="Segment",
        title="Sales by Customer Segment",
        text_auto=".2s"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # KPI Summary
    # ---------------------------------------
    st.subheader("📈 Key Performance Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Highest Monthly Sales",
        f"${monthly_sales['Sales'].max():,.0f}"
    )

    col2.metric(
        "Average Monthly Sales",
        f"${monthly_sales['Sales'].mean():,.0f}"
    )

    col3.metric(
        "Total Categories",
        df["Category"].nunique()
    )

    # ---------------------------------------
    # Business Insights
    # ---------------------------------------
    st.markdown("## 📌 Business Insights")

    top_category = category_sales.iloc[0]["Category"]
    top_region = region_sales.sort_values(
        "Sales",
        ascending=False
    ).iloc[0]["Region"]

    st.info(f"""
• Sales exhibit clear seasonal patterns.

• **{top_category}** generates the highest revenue.

• **{top_region}** is the best-performing region.

• Monthly sales fluctuate due to seasonal demand.

• Businesses should increase inventory before peak sales periods.

• Regional demand analysis can improve inventory planning and logistics.

• Product category performance can guide marketing investments.
""")
# ====================================================
# FORECAST PAGE
# ====================================================
elif page == "Forecast":

    st.header("📈 Sales Forecasting using Prophet")

    st.write("""
This dashboard forecasts future monthly sales using the Prophet forecasting model.

The forecast is generated from historical monthly sales data and includes future predictions with confidence intervals.

Three forecasting models (SARIMA, Prophet, and XGBoost) were evaluated during model development. Prophet achieved the best forecasting accuracy and was selected for deployment.
""")

    # ---------------------------------------
    # Prepare Monthly Sales Data
    # ---------------------------------------
    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales.columns = ["ds", "y"]

    # ---------------------------------------
    # Forecast Horizon
    # ---------------------------------------
    forecast_months = st.slider(
        "Select Forecast Horizon (Months)",
        min_value=3,
        max_value=24,
        value=12
    )

    # ---------------------------------------
    # Prophet Model
    # ---------------------------------------
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(monthly_sales)

    future = model.make_future_dataframe(
        periods=forecast_months,
        freq="ME"
    )

    forecast = model.predict(future)

    # ---------------------------------------
    # Forecast Data
    # ---------------------------------------
    st.subheader("Forecast Data")

    st.dataframe(
        forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].tail(forecast_months),
        use_container_width=True
    )

    # ---------------------------------------
    # Forecast Plot
    # ---------------------------------------
    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Forecasted Monthly Sales"
    )

    fig.add_scatter(
        x=monthly_sales["ds"],
        y=monthly_sales["y"],
        mode="lines",
        name="Historical Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Confidence Interval
    # ---------------------------------------
    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Forecast with Confidence Interval"
    )

    fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        name="Upper Bound"
    )

    fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        name="Lower Bound"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Prophet Components
    # ---------------------------------------
    st.subheader("Trend Components")

    component_fig = model.plot_components(forecast)

    st.pyplot(component_fig)

    # ---------------------------------------
    # KPI Cards
    # ---------------------------------------
    latest_actual = monthly_sales["y"].iloc[-1]

    forecast_value = forecast["yhat"].iloc[-1]

    growth = (
        (forecast_value - latest_actual)
        / latest_actual
    ) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Latest Sales",
        f"${latest_actual:,.0f}"
    )

    c2.metric(
        "Forecast Sales",
        f"${forecast_value:,.0f}"
    )

    c3.metric(
        "Expected Growth",
        f"{growth:.2f}%"
    )

    # ---------------------------------------
    # Business Interpretation
    # ---------------------------------------
    st.subheader("Business Interpretation")

    if growth > 10:
        st.success(
            "Sales are expected to increase significantly. Businesses should prepare for higher inventory requirements."
        )

    elif growth > 0:
        st.info(
            "Sales are expected to grow steadily over the forecast horizon."
        )

    else:
        st.warning(
            "Sales may decline. Inventory optimization and marketing strategies should be reviewed."
        )

    st.write("""
The Prophet model captures long-term trends and seasonal patterns in historical sales data.

The shaded confidence interval represents the expected range of future sales values.

Organizations can use these forecasts for demand planning, budgeting, procurement, and inventory optimization.
""")

    # ---------------------------------------
    # Download Forecast
    # ---------------------------------------
    csv = forecast.to_csv(index=False)

    st.download_button(
        "📥 Download Forecast CSV",
        csv,
        file_name="forecast.csv",
        mime="text/csv"
    )

    # ---------------------------------------
    # Model Comparison
    # ---------------------------------------
    st.subheader("📊 Forecast Model Comparison")

    comparison_df = pd.DataFrame({

        "Model": [
            "SARIMA",
            "Prophet",
            "XGBoost"
        ],

        "MAE": [
            13455.424213,
            10128.555987,
            10544.241162
        ],

        "RMSE": [
            15938.986007,
            14561.386189,
            13504.963707
        ],

        "MAPE (%)": [
            22.020378,
            14.326710,
            15.046088
        ]
    })

    st.dataframe(
        comparison_df.style.highlight_min(
            subset=["MAE", "RMSE", "MAPE (%)"],
            color="lightgreen"
        ),
        use_container_width=True
    )

    best_model = comparison_df.loc[
        comparison_df["MAPE (%)"].idxmin(),
        "Model"
    ]

    st.success(
        f"✅ {best_model} achieved the best forecasting accuracy and has been selected for deployment."
    )
# ====================================================
# ANOMALY DETECTION PAGE
# ====================================================
elif page == "Anomaly Detection":

    st.header("🚨 Sales Anomaly Detection")

    st.write("""
This section identifies unusual sales behaviour using **Isolation Forest**.

Anomalies help businesses identify unexpected sales spikes or drops caused by
festivals, promotions, supply issues, or abnormal customer demand.
""")

    # ---------------------------------------
    # Prepare Monthly Sales Data
    # ---------------------------------------
    monthly_anomaly = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    # ---------------------------------------
    # Z-Score
    # ---------------------------------------
    monthly_anomaly["Z-Score"] = zscore(monthly_anomaly["Sales"])

    monthly_anomaly["Status"] = np.where(
        abs(monthly_anomaly["Z-Score"]) > 3,
        "Anomaly",
        "Normal"
    )

    # ---------------------------------------
    # Isolation Forest
    # ---------------------------------------
    iso_model = IsolationForest(
        contamination=0.08,
        random_state=42
    )

    monthly_anomaly["Isolation Forest"] = iso_model.fit_predict(
        monthly_anomaly[["Sales"]]
    )

    monthly_anomaly["Isolation Forest"] = (
        monthly_anomaly["Isolation Forest"]
        .map({1: "Normal", -1: "Anomaly"})
    )

    # ---------------------------------------
    # Monthly Sales Table
    # ---------------------------------------
    st.subheader("Monthly Sales Data")

    st.dataframe(
        monthly_anomaly,
        use_container_width=True
    )

    # ---------------------------------------
    # Scatter Plot
    # ---------------------------------------
    fig = px.scatter(
        monthly_anomaly,
        x="Order Date",
        y="Sales",
        color="Isolation Forest",
        title="Isolation Forest Anomaly Detection",
        hover_data=["Sales", "Z-Score"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------
    # KPI Cards
    # ---------------------------------------
    total_anomalies = len(
        monthly_anomaly[
            monthly_anomaly["Isolation Forest"] == "Anomaly"
        ]
    )

    total_normal = len(
        monthly_anomaly[
            monthly_anomaly["Isolation Forest"] == "Normal"
        ]
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Detected Anomalies",
        total_anomalies
    )

    c2.metric(
        "Normal Months",
        total_normal
    )

    # ---------------------------------------
    # Business Interpretation
    # ---------------------------------------
    st.subheader("📌 Business Interpretation")

    st.info("""
Most detected anomalies correspond to unusually high or unusually low sales.

Possible reasons include:

• Festival or holiday demand

• Promotional campaigns

• Product launches

• Supply chain disruptions

• Inventory shortages

• Unexpected customer demand

• Data quality issues
""")

    # ---------------------------------------
    # Recommendations
    # ---------------------------------------
    st.subheader("💡 Business Recommendations")

    st.success("""
✔ Investigate months with unusually high sales to understand successful business strategies.

✔ Monitor low-sales anomalies to identify operational or supply chain issues.

✔ Prepare inventory before seasonal demand spikes.

✔ Combine anomaly detection with forecasting for better planning.

✔ Continuously monitor sales behaviour for early warning signs.
""")
# ====================================================
# PRODUCT SEGMENTATION PAGE
# ====================================================
elif page == "Product Segmentation":

    st.header("🎯 Product Demand Segmentation")

    st.write("""
This section groups products with similar sales behaviour using
**K-Means Clustering**.

The objective is to identify high-performing, moderate-performing,
and low-performing products to support inventory planning,
marketing, and business decision-making.
""")

    # ---------------------------------------
    # Product Summary
    # ---------------------------------------
    product_summary = (
        df.groupby("Product Name")
        .agg({
            "Sales": "sum",
            "Order ID": "nunique"
        })
        .rename(columns={"Order ID": "Orders"})
        .reset_index()
    )

    # ---------------------------------------
    # Feature Scaling
    # ---------------------------------------
    features = product_summary[["Sales", "Orders"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # ---------------------------------------
    # K-Means Clustering
    # ---------------------------------------
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    product_summary["Cluster"] = kmeans.fit_predict(scaled_features)

    # ---------------------------------------
    # Rename Clusters
    # ---------------------------------------
    cluster_names = {
        0: "⭐ High-Value Products",
        1: "📈 Moderate Performers",
        2: "📉 Low-Performing Products"
    }

    product_summary["Cluster Name"] = (
        product_summary["Cluster"]
        .map(cluster_names)
    )

    # ---------------------------------------
    # PCA Visualization
    # ---------------------------------------
    pca = PCA(n_components=2)

    points = pca.fit_transform(scaled_features)

    plot_df = pd.DataFrame(
        points,
        columns=["PC1", "PC2"]
    )

    plot_df["Cluster"] = (
        product_summary["Cluster Name"]
    )

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="Product Segmentation using PCA",
        hover_name=product_summary["Product Name"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------
    # Cluster Summary
    # ---------------------------------------
    cluster_summary = (
        product_summary
        .groupby("Cluster Name")[["Sales", "Orders"]]
        .mean()
        .round(2)
        .reset_index()
    )

    st.subheader("📊 Cluster Summary")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    # ---------------------------------------
    # Cluster Selection
    # ---------------------------------------
    selected_cluster = st.selectbox(
        "Select Cluster",
        list(cluster_names.values())
    )

    filtered = product_summary[
        product_summary["Cluster Name"] == selected_cluster
    ]

    st.dataframe(
        filtered,
        use_container_width=True
    )

    # ---------------------------------------
    # Cluster Interpretation
    # ---------------------------------------
    st.subheader("📌 Cluster Interpretation")

    if selected_cluster == "⭐ High-Value Products":

        st.success("""
### ⭐ High-Value Products

• Highest sales

• Highest customer demand

• Highest order frequency

**Recommendation**

- Maintain inventory

- Prioritize marketing

- Avoid stock shortages

- Launch premium campaigns
""")

    elif selected_cluster == "📈 Moderate Performers":

        st.info("""
### 📈 Moderate Performers

• Stable sales

• Average customer demand

• Good growth opportunity

**Recommendation**

- Cross-selling

- Product bundling

- Promotional discounts

- Seasonal campaigns
""")

    else:

        st.warning("""
### 📉 Low-Performing Products

• Low sales

• Low demand

• Low order frequency

**Recommendation**

- Review pricing

- Improve visibility

- Bundle with popular products

- Consider discontinuation if performance remains low
""")

    # ---------------------------------------
    # Download Cluster Data
    # ---------------------------------------
    csv = product_summary.to_csv(index=False)

    st.download_button(
        "📥 Download Cluster Data",
        csv,
        file_name="product_clusters.csv",
        mime="text/csv"
    )

    # ---------------------------------------
    # Dashboard Features
    # ---------------------------------------
    st.success("""
### ✅ Dashboard Features

✔ Sales Analysis

✔ Prophet Forecasting

✔ Forecast Model Comparison

✔ Anomaly Detection

✔ Product Segmentation

✔ Business Intelligence Dashboard

Developed using:

- Python

- Streamlit

- Prophet

- Plotly

- Scikit-learn
""")

    # ---------------------------------------
    # Conclusion
    # ---------------------------------------
    st.markdown("---")

    st.subheader("✅ Conclusion")

    st.write("""
This dashboard integrates **sales analytics, forecasting,
anomaly detection, and product segmentation**
to support business decision-making.

Three forecasting models (**SARIMA, Prophet, and XGBoost**) were
evaluated. **Prophet** achieved the best overall forecasting
performance and was selected for deployment.

The dashboard enables organizations to:

- Forecast future demand
- Detect unusual sales behaviour
- Identify high-performing products
- Improve inventory planning
- Support data-driven business decisions
""")