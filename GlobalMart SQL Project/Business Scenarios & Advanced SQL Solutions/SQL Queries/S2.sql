WITH Monthly AS (
    SELECT d.year, d.month, SUM(fs.total_sales) AS rev
    FROM fact_sales fs JOIN dim_date d ON fs.date_id = d.date_id
    GROUP BY d.year, d.month
)
SELECT year, month, rev, LAG(rev, 1) OVER(PARTITION BY month ORDER BY year) AS prev_rev,
ROUND(((rev - LAG(rev, 1) OVER(PARTITION BY month ORDER BY year)) / LAG(rev, 1) OVER(PARTITION BY month ORDER BY year)) * 100, 2) AS yoy_pct
FROM Monthly
ORDER by year;
