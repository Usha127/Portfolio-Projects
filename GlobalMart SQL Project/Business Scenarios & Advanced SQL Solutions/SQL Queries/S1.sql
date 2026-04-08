WITH RFM_Base AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT(transaction_id)) AS frequency,
		MAX(fs.date_id) AS max_date,
        SUM(total_sales) AS monetary
    FROM fact_sales fs
    JOIN dim_date d ON fs.date_id = d.date_id
    GROUP BY customer_id
),
RFM_Scores AS (
    SELECT 
        customer_id,
        NTILE(5) OVER (ORDER BY max_date ASC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM RFM_Base
)
SELECT 
    customer_id,
    r_score, f_score, m_score,
    CONCAT(r_score, f_score, m_score) AS rfm_cell
FROM RFM_Scores
ORDER BY customer_id