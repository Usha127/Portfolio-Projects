SELECT d.year, d.month, SUM(fs.total_sales) AS sales,
SUM(SUM(fs.total_sales)) OVER (ORDER BY d.year, d.month) AS running_total,
AVG(SUM(fs.total_sales)) OVER (ORDER BY d.year, d.month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_avg
FROM fact_sales fs JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY d.year, d.month;