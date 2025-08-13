import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Retail Forecasting", layout="wide")
st.title("🛍️ Retail Demand Forecasting App")

# Intro and Instructions
st.markdown("""
Welcome to the **Retail Demand Forecasting Tool**!

This app helps you:
- 📊 Analyze historical sales by country
- 🔮 Forecast future sales using a simple trend model
- 💡 Understand total revenue and units sold over time

---

### 📤 Step 1: Upload Your CSV File

Expected columns:
- `InvoiceDate` (e.g., 2024-01-01 10:15)
- `Quantity` (number of units sold)
- `UnitPrice` (price per unit)
- `Country` (customer's country)



""")

# File uploader
uploaded_file = st.file_uploader("Upload your retail sales CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df.dropna(inplace=True)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSale'] = df['Quantity'] * df['UnitPrice']

    # Select country
    st.subheader("🌍 Choose Country for Analysis")
    country = st.selectbox("Country", sorted(df['Country'].unique()))
    df = df[df['Country'] == country]

    # Aggregate daily total sales
    daily_sales = df.groupby('InvoiceDate')['TotalSale'].sum().resample('D').sum().fillna(0).reset_index()
    daily_sales.columns = ['date', 'daily_total_sales']

    # Aggregate daily units sold
    daily_units = df.groupby('InvoiceDate')['Quantity'].sum().resample('D').sum().fillna(0).reset_index()
    daily_units.columns = ['date', 'daily_units_sold']

    # Summary stats
    st.subheader("📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Historical Sales", f"£{daily_sales['daily_total_sales'].sum():,.2f}")
    col2.metric("Average Daily Sales", f"£{daily_sales['daily_total_sales'].mean():,.2f}")
    col3.metric("Total Units Sold", f"{int(daily_units['daily_units_sold'].sum()):,} units")

    # Time series charts
    st.subheader("📈 Historical Trends")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily Sales (£)**")
        st.line_chart(daily_sales.set_index('date'))
    with col2:
        st.markdown("**Daily Units Sold**")
        st.line_chart(daily_units.set_index('date'))

    # Prepare for forecasting
    daily_sales['day_number'] = np.arange(len(daily_sales))
    X = daily_sales[['day_number']]
    y = daily_sales['daily_total_sales']

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("🔮 Forecast Configuration")
    forecast_days = st.slider("Select number of days to forecast", 7, 90, 30)

    # Forecast future sales
    future_day_numbers = np.arange(len(daily_sales), len(daily_sales) + forecast_days).reshape(-1, 1)
    forecasted_sales = model.predict(future_day_numbers)

    last_date = daily_sales['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_sales': forecasted_sales})
    forecast_df['forecasted_sales'] = forecast_df['forecasted_sales'].clip(lower=0).round(2)

    # Plot
    st.subheader("📉 Sales Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_sales['date'], daily_sales['daily_total_sales'], label='Historical Sales')
    ax.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='Forecasted Sales', linestyle='--')
    ax.set_title('Forecasted Daily Sales')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales (£)')
    ax.legend()
    st.pyplot(fig)

    # Calculate TotalSales
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    st.title("Retail Sales Visualization")

    # Pie chart: Top 5 products by sales
    product_sales = df.groupby('Description')['TotalSales'].sum().sort_values(ascending=False)
    top_products = product_sales.head(5)

    fig, ax = plt.subplots()
    ax.pie(top_products, labels=top_products.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.subheader("Sales Distribution by Top 5 Products")
    st.pyplot(fig)
    
    # ---------------------- Bar Chart: Top 10 Products by Revenue ----------------------
    st.subheader("Top 10 Products by Sales Revenue")

    # Group by product description
    product_sales = df.groupby('Description')['TotalSales'].sum().sort_values(ascending=False)
    top_products = product_sales.head(10)

    # Plot bar chart
    fig2, ax2 = plt.subplots()
    top_products.plot(kind='barh', ax=ax2, color='skyblue')
    ax2.set_xlabel('Revenue')
    ax2.set_ylabel('Product')
    ax2.set_title('Top 10 Products by Sales')
    ax2.invert_yaxis()  # Highest on top
    st.pyplot(fig2)
    
    # Forecasted table
    st.subheader("🧾 Forecast Table")
    st.dataframe(forecast_df.rename(columns={
        'date': 'Date',
        'forecasted_sales': 'Forecasted Sales (GBP)'
    }))

    # Forecast explanation
    with st.expander("ℹ️ How This Forecast Works"):
        st.markdown("""
This forecast uses a **Linear Regression** model to project future sales trends based on historical data.

- 📈 It fits a straight line to your sales data over time.
- ❌ It does **not** consider seasonality, holidays, or external factors.
- ✅ It's fast and simple for basic demand forecasting.

For more advanced models, consider tools like **ARIMA**, **Prophet**, or **machine learning models**.
        """)

else:
    st.warning("👆 Please upload a CSV file to begin.")
