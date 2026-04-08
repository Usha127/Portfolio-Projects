SELECT p.product_name
FROM dim_products p
LEFT JOIN (
    SELECT DISTINCT product_id
    FROM fact_sales fs
    JOIN dim_date d ON fs.date_id = d.date_id
    WHERE d.full_date >= DATEADD(MONTH, -6, CAST(GETDATE() AS date))
) sales_6m
    ON p.product_id = sales_6m.product_id
WHERE sales_6m.product_id IS NULL;