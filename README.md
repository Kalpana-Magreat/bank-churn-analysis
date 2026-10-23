[README.md](https://github.com/user-attachments/files/33879487/README.md)
# Bank Customer Churn Analysis and Prediction

**Tools:** Python (pandas, scikit-learn, matplotlib) | SQL (SQLite) | Power BI

An end-to-end analytics project that finds why bank customers leave, predicts which customers are likely to leave next, and turns the results into a retention action list for the business.

---

## 1. Business problem

Acquiring a new customer costs far more than keeping an existing one. This project answers three questions for a retail bank:

1. **How big is the churn problem?** (and where is it concentrated)
2. **Why are customers leaving?** (which segments and behaviours drive churn)
3. **Who should the bank contact first?** (a prioritised list of at-risk, high-value customers)

## 2. Dataset

- **Source:** Bank Customer Churn Prediction dataset (Kaggle) - `<add link here>`
- **Size:** 10,000 customers, 12 columns
- **Fields:** credit score, country (France, Germany, Spain), gender, age, tenure, balance, number of products, credit card, active member, estimated salary, and `churn` (1 = left the bank)
- **Data quality:** no missing values, no duplicate rows or customer IDs
- **Overall churn rate: 20.4%** (2,037 of 10,000 customers)

## 3. Approach

| Step | Tool | What was done |
|---|---|---|
| 1. Cleaning and EDA | Python | Data checks, feature engineering (age groups, credit bands, zero-balance flag), churn rate by segment |
| 2. Prediction | Python (scikit-learn) | Logistic Regression and Random Forest, stratified train/test split, class balancing, evaluation with precision, recall and ROC-AUC |
| 3. Business queries | SQL (SQLite) | 10 queries using GROUP BY, CASE WHEN, subqueries, CTEs and window functions (RANK, SUM OVER) |
| 4. Dashboard | Power BI | 2-page dashboard with DAX measures, risk bands and a priority call list |

## 4. Key findings

| Finding | Evidence |
|---|---|
| **Germany churns at about twice the rate** of France and Spain | 32.4% vs 16.2% (France) and 16.7% (Spain) |
| **Churn rises sharply with age** | 7.5% (18-30), 12.1% (31-40), 34.0% (41-50), 56.2% (51-60) |
| **Number of products is a strong signal** | 2 products: 7.6% churn; 1 product: 27.7%; 3 products: 82.7%; 4 products: 100% (small groups) |
| **Inactive members churn almost twice as much** | 26.9% vs 14.3% for active members |
| **Highest-risk segment** | Inactive customers aged 41-60 in Germany churn at about 64%, vs about 18% for all other customers |
| **Women churn more than men** | 25.1% vs 16.5% |
| **Credit score, salary, tenure, and credit card ownership are weak drivers** | Churn stays roughly 19-22% across their groups |

## 5. Model results

Churners are only about 20% of customers, so accuracy alone is misleading: a model that predicts "nobody churns" is about 80% accurate but catches no one. Models were judged on recall, precision and ROC-AUC on a held-out test set (20%).

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | about 0.78 | Misses the non-linear effect of the number of products |
| **Random Forest** | **about 0.86** | Recall about 0.74, precision about 0.51 |

- **Top drivers (Random Forest):** age, number of products, balance, being in Germany, activity status.
- **Risk bands:** each customer gets a churn probability and a Low / Medium / High band (scored with 5-fold out-of-fold predictions). The actual churn rate rises sharply across bands (roughly 4% in Low vs about 63% in High), so the bands separate risk well.

## 6. Dashboard

**Page 1: Customer Churn Overview**
`<add screenshot: dashboard/page1_overview.png>`

**Page 2: Churn Drivers and Action**
`<add screenshot: dashboard/page2_drivers_action.png>`

The dashboard includes KPI cards, churn rate by country, age and gender, product and activity breakdowns, a country-by-activity heatmap, high-risk and balance-at-risk cards, and a **priority call list** of high-balance customers flagged High risk who have not left yet.

## 7. Recommendations

1. **Target inactive customers aged 41-60 in Germany** with a retention programme and personal outreach.
2. **Investigate customers holding 3 or more products.** They churn at 83% to 100%, which may point to bundling or service issues.
3. **Re-engage inactive members** with offers, reminders and app engagement.
4. **Contact the priority call list each month**, starting with high-balance customers at High risk.

## 8. Project structure

```
bank-churn-analysis/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- Bank_Customer_Churn_Prediction.csv   (raw data, see Dataset section)
|   |-- churn_clean.csv                      (output of step 1)
|   |-- churn_risk_scores.csv                (output of step 2)
|   |-- 01_churn_eda.py
|   |-- 02_churn_model.py
|   |-- 03_run_sql.py
|   |-- 03_churn_queries.sql
|-- dashboard/
|   |-- bank_churn_dashboard.pbix
|   |-- page1_overview.png
|   |-- page2_drivers_action.png
```

## 9. How to run

```bash
pip install -r requirements.txt
cd data
python 01_churn_eda.py      # cleaning + EDA -> churn_clean.csv, charts
python 02_churn_model.py    # model -> churn_risk_scores.csv, model_charts.png
python 03_run_sql.py        # SQL queries -> churn.db, sql_results.txt
```

Open `dashboard/bank_churn_dashboard.pbix` in Power BI Desktop to view the dashboard.

## 10. Limitations and next steps

- The data is a **single snapshot with no dates**, so churn over time cannot be studied.
- Findings show **association, not cause**. A retention campaign would need an A/B test to prove it works.
- Some groups are small (for example, customers with 4 products: 60 people), so those rates should be treated carefully.
- **Next steps:** add time-based data, tune the model threshold using the cost of losing a customer versus the cost of a retention offer, and test other models such as gradient boosting.

## Author

Kalpana Magreat A - Final-year B.E. Electronics and Communication Engineering
`<linkedin.com/in/kalpanamagreat>` | `<kalpuamul2023@gmail.com>`
