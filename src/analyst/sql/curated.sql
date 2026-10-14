-- Curated tables the model queries (D1.20). The gateway injects them as WITH
-- clauses, so the model only ever writes FROM sales, customers or products.
-- Blocked PII columns (D1.3) are never selected. Age is banded and sign-up
-- time is kept by month only (D1.6). Order dates come from orders (A2).
-- LEFT JOINs keep every item row, so sales reconciles with order_items.

-- name: sales
SELECT
  oi.order_id,
  oi.user_id AS customer_id,
  oi.product_id,
  oi.status,
  oi.sale_price,
  p.cost,
  p.name AS product_name,
  p.brand,
  p.category,
  p.department,
  o.created_at AS order_created_at,
  o.shipped_at AS order_shipped_at,
  o.delivered_at AS order_delivered_at,
  o.returned_at AS order_returned_at,
  CASE
    WHEN u.age < 18 THEN 'under 18'
    WHEN u.age < 25 THEN '18-24'
    WHEN u.age < 35 THEN '25-34'
    WHEN u.age < 45 THEN '35-44'
    WHEN u.age < 55 THEN '45-54'
    WHEN u.age < 65 THEN '55-64'
    ELSE '65+'
  END AS age_band,
  u.gender,
  u.country,
  u.state,
  u.city,
  u.traffic_source,
  DATE_TRUNC(DATE(u.created_at), MONTH) AS signup_month
FROM `{dataset}.order_items` AS oi
LEFT JOIN `{dataset}.orders` AS o ON o.order_id = oi.order_id
LEFT JOIN `{dataset}.products` AS p ON p.id = oi.product_id
LEFT JOIN `{dataset}.users` AS u ON u.id = oi.user_id

-- name: customers
SELECT
  u.id AS customer_id,
  CASE
    WHEN u.age < 18 THEN 'under 18'
    WHEN u.age < 25 THEN '18-24'
    WHEN u.age < 35 THEN '25-34'
    WHEN u.age < 45 THEN '35-44'
    WHEN u.age < 55 THEN '45-54'
    WHEN u.age < 65 THEN '55-64'
    ELSE '65+'
  END AS age_band,
  u.gender,
  u.country,
  u.state,
  u.city,
  u.traffic_source,
  DATE_TRUNC(DATE(u.created_at), MONTH) AS signup_month
FROM `{dataset}.users` AS u

-- name: products
SELECT
  p.id AS product_id,
  p.name AS product_name,
  p.brand,
  p.category,
  p.department,
  p.cost,
  p.retail_price
FROM `{dataset}.products` AS p
