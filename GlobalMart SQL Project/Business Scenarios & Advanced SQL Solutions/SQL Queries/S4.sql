WITH Prod_Sales AS (
    SELECT product_id, SUM(total_sales) AS sales
    FROM fact_sales GROUP BY product_id
),
Cum_Pct AS (
    SELECT product_id, sales,
    SUM(sales) OVER (ORDER BY sales DESC) / SUM(sales) OVER () AS running_pct
    FROM Prod_Sales
)
--SELECT product_id as 'Products performing under 80% of sales', Sales, Running_pct FROM Cum_Pct WHERE running_pct <= 0.85;
SELECT product_id as 'Products performing above 80% of sales', Sales, Running_pct FROM Cum_Pct WHERE running_pct >= 0.85;