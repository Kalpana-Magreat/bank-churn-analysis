import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "Bank Customer Churn Prediction.csv"

df = pd.read_csv(FILE)
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customer_id"].duplicated().sum())
print("Overall churn rate: %.1f%%" % (df["churn"].mean() * 100))

df["age_group"] = pd.cut(df["age"], [17, 30, 40, 50, 60, 100],
                         labels=["18-30", "31-40", "41-50", "51-60", "60+"])
df["credit_band"] = pd.cut(df["credit_score"], [0, 579, 669, 739, 799, 850],
                           labels=["Poor", "Fair", "Good", "Very Good", "Excellent"])
df["zero_balance"] = np.where(df["balance"] == 0, "Zero balance", "Has balance")
df["salary_band"] = pd.qcut(df["estimated_salary"], 4, labels=["Q1 Low", "Q2", "Q3", "Q4 High"])

def churn_by(col):
    g = df.groupby(col, observed=True)["churn"].agg(rate="mean", customers="count")
    g["rate"] = (g["rate"] * 100).round(1)
    return g

segments = ["country", "gender", "age_group", "products_number", "active_member",
            "zero_balance", "credit_band", "salary_band"]
for s in segments:
    print(f"\nChurn % by {s}")
    print(churn_by(s).to_string())

print("\nChurn % by country x active_member")
print((df.groupby(["country", "active_member"])["churn"].mean().unstack() * 100).round(1))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, ["country", "gender", "age_group", "products_number", "active_member", "zero_balance"]):
    g = churn_by(col)
    ax.bar(g.index.astype(str), g["rate"])
    ax.axhline(df["churn"].mean() * 100, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Churn % by {col}")
    ax.set_ylabel("Churn %")
    for i, v in enumerate(g["rate"]):
        ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("eda_churn_by_segment.png", dpi=150)

df.to_csv("churn_clean.csv", index=False)
print("\nSaved: churn_clean.csv, eda_churn_by_segment.png")
