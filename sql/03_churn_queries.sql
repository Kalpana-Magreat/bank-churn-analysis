-- Bank Customer Churn: business queries
-- Tables: customers (from churn_clean.csv), risk (from churn_risk_scores.csv)

-- Q1: Overall churn
SELECT COUNT(*) AS customers,
       SUM(churn) AS churned,
       ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers;

-- Q2: Churn and average balance by country
SELECT country,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct,
       ROUND(AVG(balance), 0) AS avg_balance
FROM customers
GROUP BY country
ORDER BY churn_rate_pct DESC;

-- Q3: Churn by age group
SELECT age_group,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
FROM customers
GROUP BY age_group
ORDER BY age_group;

-- Q4: Churn by number of products held
SELECT products_number,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
FROM customers
GROUP BY products_number
ORDER BY products_number;

-- Q5: Money lost: total balance held by churned customers, by country
SELECT country,
       SUM(CASE WHEN churn = 1 THEN balance ELSE 0 END) AS balance_lost,
       ROUND(100.0 * SUM(CASE WHEN churn = 1 THEN balance ELSE 0 END) / SUM(balance), 1) AS pct_of_country_balance
FROM customers
GROUP BY country
ORDER BY balance_lost DESC;

-- Q6: Highest-risk segment: inactive customers aged 41-60 in Germany vs everyone else
SELECT CASE WHEN country = 'Germany' AND active_member = 0 AND age BETWEEN 41 AND 60
            THEN 'Germany, inactive, 41-60' ELSE 'All other customers' END AS segment,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
FROM customers
GROUP BY segment;

-- Q7: Window function: rank age groups by churn within each country
SELECT country, age_group, churn_rate_pct,
       RANK() OVER (PARTITION BY country ORDER BY churn_rate_pct DESC) AS risk_rank
FROM (
    SELECT country, age_group, ROUND(100.0 * AVG(churn), 1) AS churn_rate_pct
    FROM customers
    GROUP BY country, age_group
)
ORDER BY country, risk_rank;

-- Q8: Model risk bands: customers, actual churn, and balance at stake
SELECT risk_band,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 1) AS actual_churn_pct,
       ROUND(SUM(balance), 0) AS total_balance
FROM risk
GROUP BY risk_band
ORDER BY CASE risk_band WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;

-- Q9: Priority call list: high-balance customers the model flags as High risk who have not left yet
SELECT r.customer_id, r.country, r.age, r.products_number, r.active_member,
       ROUND(r.balance, 0) AS balance, r.churn_probability
FROM risk r
WHERE r.risk_band = 'High' AND r.churn = 0 AND r.balance > 100000
ORDER BY r.churn_probability DESC
LIMIT 10;

-- Q10: Running share of lost balance: which countries make up the loss (CTE + window)
WITH lost AS (
    SELECT country, SUM(balance) AS balance_lost
    FROM customers
    WHERE churn = 1
    GROUP BY country
)
SELECT country,
       ROUND(balance_lost, 0) AS balance_lost,
       ROUND(100.0 * balance_lost / SUM(balance_lost) OVER (), 1) AS share_of_total_loss_pct
FROM lost
ORDER BY balance_lost DESC;
