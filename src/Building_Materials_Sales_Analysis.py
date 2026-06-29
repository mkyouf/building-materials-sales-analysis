# ==========================================================
# Building Materials Sales Analysis
# ==========================================================

# Business Problem:
# This project analyzes sales data from a building materials store
# to uncover sales trends and support business decisions.

# Tools Used:
# - Python
# - Pandas
# - NumPy
# - Matplotlib
# - Seaborn

# Author: Ali Almayouf

# ==========================================================
# Session 1 - Import Libraries
# ==========================================================

# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display Options
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Session 2 - Load Dataset
# ==========================================================

df = pd.read_csv("building_supply_store_sales_2020-2024.csv")

print("Dataset Loaded Successfully!")

# ==========================================================
# Session 3 - Data Overview
# ==========================================================

# Display the first five rows
print("\nFirst Five Rows:")
print(df.head())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical summary for numerical columns
print("\nNumerical Summary:")
print(df.describe())

# Statistical summary for categorical columns
print("\nCategorical Summary:")
print(df.describe(include='object'))

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# ==========================================================
# Session 4 - Data Cleaning
# ==========================================================

"""
Objective:
Clean the dataset by checking for missing values,
duplicate records, incorrect data types,
and preparing the data for analysis.
"""

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

duplicates = df.duplicated().sum()

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed {duplicates} duplicate rows.")
else:
    print("No duplicate rows found.")

print("Duplicates Removed:", df.duplicated().sum())

# Convert Date Column
df["date"] = pd.to_datetime(df["date"])

print(df.dtypes)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================================
# Session 5 - Exploratory Data Analysis (EDA)
# ==========================================================

"""
Objective:
Explore the dataset to identify sales patterns,
customer behavior, and business insights.
"""
# ==========================================================
# Business Question 1
# Which products generate the highest revenue?
# ==========================================================

product_sales = (
    df.groupby("product_name")["total_price"]
      .sum()
      .sort_values(ascending=False)
)

print("\nTop 10 Products by Revenue")
print(product_sales.head(10))

plt.figure(figsize=(12,6))

product_sales.head(10).plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("Top 10 Products by Revenue", fontsize=15)
plt.xlabel("Product")
plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

brand_sales = (
    df.groupby("brand")["total_price"]
      .sum()
      .sort_values(ascending=False)
)

print("\nRevenue by Brand")
print(brand_sales)

plt.figure(figsize=(10,6))

brand_sales.plot(
    kind="bar",
    color="darkorange",
    edgecolor="black"
)

plt.title("Revenue by Brand", fontsize=15)
plt.xlabel("Brand")
plt.ylabel("Revenue")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

category_sales = (
    df.groupby("product_type")["total_price"]
      .sum()
      .sort_values(ascending=False)
)

print(category_sales)

plt.figure(figsize=(10,6))

category_sales.plot(
    kind="bar",
    color="seagreen",
    edgecolor="black"
)

plt.title("Revenue by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Revenue")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

store_sales = (
    df.groupby("store_location")["total_price"]
      .sum()
      .sort_values(ascending=False)
)

print(store_sales)

plt.figure(figsize=(11,6))

store_sales.plot(
    kind="bar",
    color="royalblue",
    edgecolor="black"
)

plt.title("Revenue by Store Location")
plt.xlabel("Store")
plt.ylabel("Revenue")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

delivery = df["delivery"].value_counts()

print(delivery)

plt.figure(figsize=(7,7))

delivery.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.ylabel("")
plt.title("Delivery Method")

plt.show()

monthly_sales = (
    df.groupby(df["date"].dt.to_period("M"))["total_price"]
      .sum()
)

monthly_sales.index = monthly_sales.index.astype(str)

plt.figure(figsize=(14,6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

top_customers = (
    df.groupby("customer_name")["total_price"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_customers)

plt.figure(figsize=(12,6))

top_customers.plot(
    kind="bar",
    color="purple",
    edgecolor="black"
)

plt.title("Top 10 Customers")
plt.xlabel("Customer")
plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()

plt.figure(figsize=(10,6))

plt.hist(
    df["total_price"],
    bins=30,
    edgecolor="black"
)

plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Frequency")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# ==========================================================
# Session 6 - Feature Engineering
# ==========================================================

"""
Objective:
Create new features that provide deeper business insights
and improve the quality of future analysis.
"""

# ----------------------------------------------------------
# Date Features
# ----------------------------------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month_name()
df["month_number"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()
df["quarter"] = df["date"].dt.quarter

# Weekend Indicator
df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"])

# ==========================================================
# Profit Per Unit
# ==========================================================

df["profit_per_unit"] = (
    df["price_per_unit"] - df["unit_purchase"]
)

# ==========================================================
# Profit Margin
# ==========================================================

df["profit_margin"] = (
    (df["profit_per_unit"] / df["price_per_unit"]) * 100
).round(2)

monthly_profit = (
    df.groupby("month")["profit_per_unit"]
      .mean()
      .sort_values(ascending=False)
)

print(monthly_profit)

brand_profit = (
    df.groupby("brand")["profit_per_unit"]
      .mean()
      .sort_values(ascending=False)
)

print(brand_profit)

weekend_sales = (
    df.groupby("is_weekend")["total_price"]
      .sum()
)

print(weekend_sales)

# ==========================================================
# Session 7 - Correlation Analysis
# ==========================================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

print(numeric_df.head())

correlation = numeric_df.corr()

print(correlation)

plt.figure(figsize=(10,7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

# ==========================================================
# Session 8 - Business Insights
# ==========================================================

"""
Key Business Insights

1. High-priced products contribute significantly to total revenue.

2. Profit margin remains relatively stable across products.

3. Store performance varies by location,
   indicating opportunities for inventory optimization.

4. Delivery and pickup services both contribute
   significantly to total sales.

5. Monthly sales trends reveal seasonal demand,
   helping management improve purchasing decisions.
"""
df.to_csv("building_supply_store_sales_cleaned.csv", index=False)

print("New CSV Saved Successfully!")