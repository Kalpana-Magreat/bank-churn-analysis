import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

df = pd.read_csv("churn_clean.csv")

features = ["credit_score", "country", "gender", "age", "tenure", "balance",
            "products_number", "credit_card", "active_member", "estimated_salary"]
X = pd.get_dummies(df[features], drop_first=True)  
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1),
}

base_pred = np.zeros(len(y_test), dtype=int)
rows = [["Baseline (nobody churns)", accuracy_score(y_test, base_pred),
         0.0, 0.0, 0.0, 0.5]]

fitted, probs = {}, {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    fitted[name], probs[name] = m, p
    rows.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred),
                 recall_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, p)])

results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]).round(3)
print("\nModel comparison (test set):\n", results.to_string(index=False))

best_name = results.iloc[1:].sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
best = fitted[best_name]
print("\nBest model:", best_name)

pred_best = (probs[best_name] >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_best).ravel()
print(f"Confusion matrix -> caught churners: {tp}, missed churners: {fn}, "
      f"false alarms: {fp}, correctly kept: {tn}")

rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom forest feature importance:\n", imp.round(3).head(8))

lr = fitted["Logistic Regression"]
coef = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\nLogistic regression coefficients (standardised; + raises churn risk):\n", coef.round(2).head(8))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
risk = cross_val_predict(models["Random Forest"], X, y, cv=cv, method="predict_proba")[:, 1]
out = df[["customer_id", "country", "gender", "age", "age_group", "balance",
          "products_number", "active_member", "churn"]].copy()
out["churn_probability"] = risk.round(3)
out["risk_band"] = pd.cut(risk, [-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
out.to_csv("churn_risk_scores.csv", index=False)
print("\nCustomers by risk band:\n", out["risk_band"].value_counts())
print("Actual churn rate within each band (%):\n",
      (out.groupby("risk_band", observed=True)["churn"].mean() * 100).round(1))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for name in probs:
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, probs[name]):.2f})")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set_title("ROC curve")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].legend()

imp.head(8).sort_values().plot.barh(ax=axes[1])
axes[1].set_title("Top drivers of churn (Random Forest)")

cm = confusion_matrix(y_test, pred_best)
axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
axes[2].set_yticks([0, 1], ["Actually stayed", "Actually churned"])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
axes[2].set_title(f"Confusion matrix ({best_name})")

plt.tight_layout()
plt.savefig("model_charts.png", dpi=150)
print("\nSaved: churn_risk_scores.csv, model_charts.png")
