"""
PostgreSQL Test Database Setup

This script sets up a test PostgreSQL database with sample e-commerce data.
Requires PostgreSQL to be installed and running locally.

OPTION 1: Using Docker (recommended)
    docker run --name test-postgres -e POSTGRES_PASSWORD=testpass -p 5432:5432 -d postgres:15
    py -3.11 scripts/setup_postgres_test.py

OPTION 2: Using existing PostgreSQL
    py -3.11 scripts/setup_postgres_test.py
"""

import os
import random
from datetime import datetime, timedelta

# Connection parameters for testing
CONNECTION_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "database": "test_ecommerce",
    "username": "postgres",
    "password": "testpass",
}


def create_database():
    """Create test database with sample data"""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    # Connect to default postgres database to create test database
    try:
        conn = psycopg2.connect(
            host=CONNECTION_PARAMS["host"],
            port=CONNECTION_PARAMS["port"],
            database="postgres",
            user=CONNECTION_PARAMS["username"],
            password=CONNECTION_PARAMS["password"],
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Drop and recreate database
        cursor.execute("DROP DATABASE IF EXISTS test_ecommerce")
        cursor.execute("CREATE DATABASE test_ecommerce")
        cursor.close()
        conn.close()
        print("✓ Created database: test_ecommerce")
    except Exception as e:
        print(f"Error creating database: {e}")
        print("\nMake sure PostgreSQL is running. Try:")
        print("  docker run --name test-postgres -e POSTGRES_PASSWORD=testpass -p 5432:5432 -d postgres:15")
        return False

    # Connect to new database and create tables
    conn = psycopg2.connect(
        host=CONNECTION_PARAMS["host"],
        port=CONNECTION_PARAMS["port"],
        database=CONNECTION_PARAMS["database"],
        user=CONNECTION_PARAMS["username"],
        password=CONNECTION_PARAMS["password"],
    )
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
        CREATE TABLE customers (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            city VARCHAR(50),
            country VARCHAR(50),
            segment VARCHAR(30),
            lifetime_value DECIMAL(10,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            category VARCHAR(50) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            cost DECIMAL(10,2) NOT NULL,
            stock INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            order_date TIMESTAMP NOT NULL,
            status VARCHAR(20) NOT NULL,
            total DECIMAL(10,2) NOT NULL,
            payment_method VARCHAR(20)
        );
        
        CREATE TABLE order_items (
            id SERIAL PRIMARY KEY,
            order_id INTEGER REFERENCES orders(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL
        );
        
        CREATE TABLE daily_metrics (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            revenue DECIMAL(12,2) DEFAULT 0,
            orders INTEGER DEFAULT 0,
            new_customers INTEGER DEFAULT 0,
            avg_order_value DECIMAL(10,2) DEFAULT 0
        );
    """
    )

    print("✓ Created tables")

    # Populate customers
    segments = ["Premium", "Standard", "Budget", "Enterprise"]
    countries = ["USA", "UK", "Germany", "France", "Canada"]
    cities = ["New York", "London", "Berlin", "Paris", "Toronto"]

    for i in range(200):
        cursor.execute(
            """
            INSERT INTO customers (name, email, city, country, segment, lifetime_value)
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (
                f"Customer {i+1}",
                f"customer{i+1}@example.com",
                random.choice(cities),
                random.choice(countries),
                random.choice(segments),
                round(random.uniform(100, 5000), 2),
            ),
        )
    print("✓ Added 200 customers")

    # Populate products
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    for i in range(50):
        cost = round(random.uniform(10, 200), 2)
        price = round(cost * random.uniform(1.3, 2.5), 2)
        cursor.execute(
            """
            INSERT INTO products (name, category, price, cost, stock)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (f"Product {i+1}", random.choice(categories), price, cost, random.randint(0, 500)),
        )
    print("✓ Added 50 products")

    # Populate orders
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    payments = ["credit_card", "paypal", "bank_transfer"]
    base_date = datetime.now() - timedelta(days=180)

    for i in range(500):
        customer_id = random.randint(1, 200)
        order_date = base_date + timedelta(days=random.randint(0, 180))
        total = round(random.uniform(20, 500), 2)

        cursor.execute(
            """
            INSERT INTO orders (customer_id, order_date, status, total, payment_method)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """,
            (customer_id, order_date, random.choice(statuses), total, random.choice(payments)),
        )

        order_id = cursor.fetchone()[0]

        for _ in range(random.randint(1, 4)):
            product_id = random.randint(1, 50)
            quantity = random.randint(1, 5)
            unit_price = round(random.uniform(10, 100), 2)
            cursor.execute(
                """
                INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                VALUES (%s, %s, %s, %s)
            """,
                (order_id, product_id, quantity, unit_price),
            )
    print("✓ Added 500 orders")

    # Populate daily_metrics
    for i in range(180):
        date = (base_date + timedelta(days=i)).date()
        revenue = round(random.uniform(1000, 10000), 2)
        orders = random.randint(10, 100)
        cursor.execute(
            """
            INSERT INTO daily_metrics (date, revenue, orders, new_customers, avg_order_value)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (date, revenue, orders, random.randint(1, 20), round(revenue / orders, 2)),
        )
    print("✓ Added 180 days of metrics")

    conn.commit()
    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print("POSTGRESQL TEST DATABASE READY")
    print("=" * 60)
    print("\nConnection Parameters:")
    print(f"  Host:     {CONNECTION_PARAMS['host']}")
    print(f"  Port:     {CONNECTION_PARAMS['port']}")
    print(f"  Database: {CONNECTION_PARAMS['database']}")
    print(f"  Username: {CONNECTION_PARAMS['username']}")
    print(f"  Password: {CONNECTION_PARAMS['password']}")
    print("\nTables:")
    print("  - customers (200 rows)")
    print("  - products (50 rows)")
    print("  - orders (500 rows)")
    print("  - order_items (~1000 rows)")
    print("  - daily_metrics (180 rows)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    create_database()
