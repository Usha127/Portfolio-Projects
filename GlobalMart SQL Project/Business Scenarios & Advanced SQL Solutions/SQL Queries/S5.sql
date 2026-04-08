WITH Ranked AS (
    SELECT p.category, p.product_name, SUM(fs.total_sales) AS revenue,
    DENSE_RANK() OVER (PARTITION BY p.category ORDER BY SUM(fs.total_sales) DESC) AS ranking
    FROM fact_sales fs JOIN dim_products p ON fs.product_id = p.product_id
    GROUP BY p.category, p.product_name
)
SELECT * FROM Ranked WHERE ranking <= 3;