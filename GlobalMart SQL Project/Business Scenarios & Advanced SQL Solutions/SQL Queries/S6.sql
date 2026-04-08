SELECT a.product_id AS prod_1, b.product_id AS prod_2, COUNT(*) AS bundles_sold
FROM fact_sales a
JOIN fact_sales b ON a.transaction_id = b.transaction_id AND a.product_id < b.product_id
GROUP BY a.product_id, b.product_id
ORDER BY bundles_sold DESC;