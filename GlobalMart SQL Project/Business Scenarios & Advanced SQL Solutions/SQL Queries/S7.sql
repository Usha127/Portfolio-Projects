SELECT
    c.customer_name,
    MAX(d.full_date) AS last_active,
    DATEDIFF(day, MAX(d.full_date), CAST(GETDATE() AS date)) AS days_since_last
FROM fact_sales fs
JOIN dim_customers c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY
    c.customer_name
HAVING
    DATEDIFF(day, MAX(d.full_date), CAST(GETDATE() AS date)) > 90;