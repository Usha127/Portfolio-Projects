-- ===========================================================================
-- GLOBALMART STAR SCHEMA - DATA SEEDING SCRIPT
-- ===========================================================================

-- 1. Dim_Customers
INSERT INTO Dim_Customers VALUES
('C1001', 'Alice Johnson', 'alice@email.com', '2024-05-12'),
('C1002', 'Bob Smith', 'bob@email.com', '2024-08-22'),
('C1003', 'Charlie Brown', 'charlie@email.com', '2025-01-05'),
('C1004', 'Diana Prince', 'diana@email.com', '2025-03-15'),
('C1005', 'Eve Adams', 'eve@email.com', '2025-04-10'); -- Added for Scenario 7 (Churn)

-- 2. Dim_Products
INSERT INTO Dim_Products VALUES
('P2001', 'Wireless Mouse', 'Electronics', 'Accessories', 25.00),
('P2002', 'Mechanical Keyboard', 'Electronics', 'Accessories', 80.00),
('P2003', 'Ergonomic Chair', 'Furniture', 'Office', 250.00),
('P2004', 'Noise Cancelling Headphones', 'Electronics', 'Audio', 150.00),
('P2005', 'Standing Desk', 'Furniture', 'Office', 400.00);

-- 3. Dim_Stores
INSERT INTO Dim_Stores VALUES
('S3001', 'Downtown MegaStore', 'North', 'New York'),
('S3002', 'Suburban Outlet', 'North', 'Boston'),
('S3003', 'Sun Valley Branch', 'South', 'Miami'),
('S3004', 'Pacific Hub', 'West', 'Los Angeles');

-- 4. Dim_Date
INSERT INTO Dim_Date VALUES
(20250115, '2025-01-15', 2025, 1, 15, 1),
(20250220, '2025-02-20', 2025, 2, 20, 1),
(20250610, '2025-06-10', 2025, 6, 10, 2),
(20251005, '2025-10-05', 2025, 10, 5, 4),
(20260112, '2026-01-12', 2026, 1, 12, 1),
(20260115, '2026-01-15', 2026, 1, 15, 1),
(20260220, '2026-02-20', 2026, 2, 20, 1),
(20260325, '2026-03-25', 2026, 3, 25, 1),
(20260401, '2026-04-01', 2026, 4, 1, 2);

-- 5. Fact_Sales (Interconnected transactions over 2025 & 2026)
INSERT INTO Fact_Sales (transaction_id, customer_id, product_id, store_id, date_id, quantity, total_sales, discount_amount) VALUES
-- 2025 Transactions
(1001, 'C1001', 'P2001', 'S3001', 20250115, 2, 50.00, 0.00),   -- Scenario 6: Bundle Item A
(1001, 'C1001', 'P2002', 'S3001', 20250115, 1, 80.00, 0.00),   -- Scenario 6: Bundle Item B
(1002, 'C1002', 'P2003', 'S3003', 20250115, 1, 225.00, 25.00), -- Scenario 8: Only sale of P2003
(1003, 'C1001', 'P2002', 'S3001', 20250220, 1, 80.00, 0.00),
(1004, 'C1005', 'P2004', 'S3002', 20250610, 1, 150.00, 0.00), -- Scenario 7: Eve Adams' ONLY purchase!
(1005, 'C1002', 'P2001', 'S3003', 20251005, 4, 100.00, 0.00),

-- 2026 Transactions
(1006, 'C1001', 'P2005', 'S3001', 20260112, 1, 400.00, 0.00), -- Scenario 2: Jan 2026 sales
(1007, 'C1003', 'P2002', 'S3004', 20260115, 2, 160.00, 0.00), -- Scenario 2: Jan 2026 sales
(1008, 'C1004', 'P2001', 'S3002', 20260220, 1, 25.00, 0.00),
(1009, 'C1004', 'P2001', 'S3002', 20260325, 1, 25.00, 0.00),   -- Scenario 6: Bundle Item X
(1009, 'C1004', 'P2004', 'S3002', 20260325, 1, 135.00, 15.00), -- Scenario 6: Bundle Item Y
(1010, 'C1002', 'P2002', 'S3003', 20260401, 1, 80.00, 0.00);