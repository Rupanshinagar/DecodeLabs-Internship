import pandas as pd
import numpy as np

df = pd.read_excel("Dataset for Data Analytics.xlsx")

print("Original shape:", df.shape)
print(df.head())

print("\nMissing values before:\n", df.isnull().sum())

df["CouponCode"] = df["CouponCode"].fillna("NoCoupon")

print("\nMissing values after:\n", df.isnull().sum())

numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"{col}: {outlier_count} outliers found (bounds: {lower_bound:.2f} to {upper_bound:.2f})")

    df[col] = df[col].clip(lower_bound, upper_bound)

df["OrderYear"] = df["Date"].dt.year
df["OrderMonth"] = df["Date"].dt.month
df["OrderDayOfWeek"] = df["Date"].dt.day_name()

df["AvgItemPrice"] = df["TotalPrice"] / df["Quantity"]

df["HasCoupon"] = (df["CouponCode"] != "NoCoupon").astype(int)

df["CartUtilization"] = df["Quantity"] / df["ItemsInCart"]

print("\nNew columns added:")
print(df[["OrderYear", "OrderMonth", "OrderDayOfWeek",
          "AvgItemPrice", "HasCoupon", "CartUtilization"]].head())

categorical_cols = ["PaymentMethod", "OrderStatus", "ReferralSource"]

df_encoded = pd.get_dummies(df, columns=categorical_cols)

print("\nShape after encoding:", df_encoded.shape)

df_encoded.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_dataset.csv")