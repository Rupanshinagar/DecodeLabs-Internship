import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_excel("Dataset for Data Analytics.xlsx")

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

df["CouponCode"] = df["CouponCode"].fillna("No Coupon")

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["MonthName"] = df["Date"].dt.strftime("%b-%Y")

df["ExpectedPrice"] = df["Quantity"] * df["UnitPrice"]
df["PriceDiff"] = df["TotalPrice"] - df["ExpectedPrice"]
print("\nRows jahan TotalPrice mismatch hai:")
print((df["PriceDiff"].abs() > 1).sum(), "rows")

dup_customers = df["CustomerID"].value_counts()
print("\nRepeat customers (>1 order):", (dup_customers > 1).sum())

print("\n===== Numeric Summary =====")
print(df[["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]].describe())

revenue_by_product = df.groupby("Product")["TotalPrice"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
print("\n===== Revenue by Product =====\n", revenue_by_product)

status_counts = df["OrderStatus"].value_counts()
cancel_return_rate = (status_counts.get("Cancelled", 0) + status_counts.get("Returned", 0)) / len(df) * 100
print(f"\nCancellation + Return Rate: {cancel_return_rate:.2f}%")

payment_status_ct = pd.crosstab(df["PaymentMethod"], df["OrderStatus"], normalize="index") * 100
print("\n===== Payment Method vs Order Status (%) =====\n", payment_status_ct.round(1))

referral_perf = df.groupby("ReferralSource")["TotalPrice"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
print("\n===== Referral Source Performance =====\n", referral_perf)

coupon_impact = df.groupby("CouponCode")["TotalPrice"].mean().sort_values(ascending=False)
print("\n===== Avg Order Value by Coupon =====\n", coupon_impact)

monthly_revenue = df.groupby(df["Date"].dt.to_period("M"))["TotalPrice"].sum()

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

revenue_by_product["sum"].plot(kind="bar", ax=axes[0, 0], color="teal")
axes[0, 0].set_title("Total Revenue by Product")
axes[0, 0].set_ylabel("Revenue")

status_counts.plot(kind="pie", ax=axes[0, 1], autopct="%1.1f%%", startangle=90)
axes[0, 1].set_title("Order Status Distribution")
axes[0, 1].set_ylabel("")

df["PaymentMethod"].value_counts().plot(kind="bar", ax=axes[0, 2], color="orange")
axes[0, 2].set_title("Orders by Payment Method")

referral_perf["sum"].plot(kind="bar", ax=axes[1, 0], color="purple")
axes[1, 0].set_title("Revenue by Referral Source")

monthly_revenue.plot(ax=axes[1, 1], marker="o", color="green")
axes[1, 1].set_title("Monthly Revenue Trend")
axes[1, 1].tick_params(axis="x", rotation=45)

coupon_impact.plot(kind="bar", ax=axes[1, 2], color="crimson")
axes[1, 2].set_title("Avg Order Value by Coupon Code")

plt.tight_layout()
plt.savefig("ecommerce_analytics_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Dashboard saved as ecommerce_analytics_dashboard.png")

df.to_csv("cleaned_ecommerce_data.csv", index=False)
print("Cleaned dataset exported as cleaned_ecommerce_data.csv")