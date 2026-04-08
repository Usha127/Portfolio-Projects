-- ===========================================================================
-- GLOBALMART STAR SCHEMA - DDL SCRIPT
-- ===========================================================================

-- 1. Create Dimension Tables
CREATE TABLE Dim_Customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    join_date DATE
);

CREATE TABLE Dim_Products (
    product_id VARCHAR(20) PRIMARY KEY,
    product_name VARCHAR(150) NOT NULL,
    category VARCHAR(50),
    sub_category VARCHAR(50),
    unit_price DECIMAL(10, 2)
);

CREATE TABLE Dim_Stores (
    store_id VARCHAR(20) PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    city VARCHAR(50)
);

CREATE TABLE Dim_Date (
    date_id INT PRIMARY KEY, -- Format: YYYYMMDD
    full_date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    quarter INT NOT NULL
);

-- 2. Create Fact Table
CREATE TABLE Fact_Sales (
    sales_id INT IDENTITY(1,1) PRIMARY KEY,      -- Unique line item identifier
    transaction_id INT NOT NULL,      -- Groups multiple products per receipt
    customer_id VARCHAR(20),
    product_id VARCHAR(20),
    store_id VARCHAR(20),
    date_id INT,
    quantity INT NOT NULL,
    total_sales DECIMAL(12, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0.00,
    
    -- Defining Foreign Key Relationships (Cardinalities)
    CONSTRAINT fk_customer FOREIGN KEY (customer_id) REFERENCES Dim_Customers(customer_id),
    CONSTRAINT fk_product FOREIGN KEY (product_id) REFERENCES Dim_Products(product_id),
    CONSTRAINT fk_store FOREIGN KEY (store_id) REFERENCES Dim_Stores(store_id),
    CONSTRAINT fk_date FOREIGN KEY (date_id) REFERENCES Dim_Date(date_id)
);

-- Indexing for performance optimization
CREATE INDEX idx_sales_customer ON Fact_Sales(customer_id);
CREATE INDEX idx_sales_product ON Fact_Sales(product_id);
CREATE INDEX idx_sales_date ON Fact_Sales(date_id);