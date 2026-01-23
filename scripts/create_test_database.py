"""
Create a persistent test database for frontend testing.
This creates an SQLite database with sample e-commerce data.

Run with: py -3.11 scripts/create_test_database.py
"""

import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta


def create_test_database():
    """Create a test database with sample data"""

    # Create database in data/databases directory
    db_dir = Path("data/databases")
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "test_ecommerce.db"

    # Remove if exists
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            city TEXT,
            country TEXT,
            segment TEXT,
            lifetime_value REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            cost REAL NOT NULL,
            stock INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date DATETIME NOT NULL,
            status TEXT NOT NULL,
            total REAL NOT NULL,
            payment_method TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        
        CREATE TABLE daily_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            revenue REAL DEFAULT 0,
            orders INTEGER DEFAULT 0,
            new_customers INTEGER DEFAULT 0,
            avg_order_value REAL DEFAULT 0
        );
    """
    )

    # Populate customers (200 records)
    segments = ["Premium", "Standard", "Budget", "Enterprise"]
    countries = ["USA", "UK", "Germany", "France", "Canada", "Australia"]
    cities = ["New York", "London", "Berlin", "Paris", "Toronto", "Sydney"]

    for i in range(200):
        conn.execute(
            """
            INSERT INTO customers (name, email, city, country, segment, lifetime_value)
            VALUES (?, ?, ?, ?, ?, ?)
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

    # Populate products (50 records)
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]

    for i in range(50):
        cost = round(random.uniform(10, 200), 2)
        price = round(cost * random.uniform(1.3, 2.5), 2)
        conn.execute(
            """
            INSERT INTO products (name, category, price, cost, stock)
            VALUES (?, ?, ?, ?, ?)
        """,
            (f"Product {i+1}", random.choice(categories), price, cost, random.randint(0, 500)),
        )

    # Populate orders (500 records)
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    payments = ["credit_card", "paypal", "bank_transfer"]
    base_date = datetime.now() - timedelta(days=180)

    for i in range(500):
        customer_id = random.randint(1, 200)
        order_date = base_date + timedelta(days=random.randint(0, 180))
        total = round(random.uniform(20, 500), 2)

        conn.execute(
            """
            INSERT INTO orders (customer_id, order_date, status, total, payment_method)
            VALUES (?, ?, ?, ?, ?)
        """,
            (customer_id, order_date.isoformat(), random.choice(statuses), total, random.choice(payments)),
        )

        # Add order items
        order_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for _ in range(random.randint(1, 4)):
            product_id = random.randint(1, 50)
            quantity = random.randint(1, 5)
            unit_price = round(random.uniform(10, 100), 2)
            conn.execute(
                """
                INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                VALUES (?, ?, ?, ?)
            """,
                (order_id, product_id, quantity, unit_price),
            )

    # Populate daily_metrics (180 days)
    for i in range(180):
        date = (base_date + timedelta(days=i)).date().isoformat()
        revenue = round(random.uniform(1000, 10000), 2)
        orders = random.randint(10, 100)

        conn.execute(
            """
            INSERT INTO daily_metrics (date, revenue, orders, new_customers, avg_order_value)
            VALUES (?, ?, ?, ?, ?)
        """,
            (date, revenue, orders, random.randint(1, 20), round(revenue / orders, 2)),
        )

    conn.commit()
    conn.close()

    print("=" * 60)
    print("TEST DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nDatabase Path: {db_path.absolute()}")
    print("\nTables created:")
    print("  - customers (200 rows)")
    print("  - products (50 rows)")
    print("  - orders (500 rows)")
    print("  - order_items (~1000 rows)")
    print("  - daily_metrics (180 rows)")
    print("\n" + "=" * 60)
    print("CONNECTION PARAMETERS FOR FRONTEND:")
    print("=" * 60)
    print(f"\nDatabase Type: SQLite")
    print(f"File Path: {db_path.absolute()}")
    print("\n" + "=" * 60)

    return db_path


if __name__ == "__main__":
    create_test_database()
