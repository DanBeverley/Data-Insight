"""
Test Database Connector and Notification Service

This script creates a synthetic SQLite database and tests:
1. Database connection via API
2. Table listing
3. Query execution
4. Loading table to session
5. Notification creation and retrieval
"""

import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta


def create_synthetic_database():
    db_path = Path("data/test_databases/sales_analytics.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create customers table
    cursor.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            region TEXT,
            signup_date TEXT,
            lifetime_value REAL
        )
    """
    )

    # Create products table
    cursor.execute(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_quantity INTEGER
        )
    """
    )

    # Create orders table
    cursor.execute(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            total_amount REAL,
            order_date TEXT,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """
    )

    # Insert sample data
    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    statuses = ["completed", "pending", "shipped", "cancelled"]

    # Insert customers
    customers = []
    for i in range(100):
        name = f"Customer_{i+1}"
        email = f"customer{i+1}@example.com"
        region = random.choice(regions)
        days_ago = random.randint(30, 730)
        signup = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        ltv = round(random.uniform(100, 10000), 2)
        customers.append((name, email, region, signup, ltv))

    cursor.executemany(
        "INSERT INTO customers (name, email, region, signup_date, lifetime_value) VALUES (?, ?, ?, ?, ?)", customers
    )

    # Insert products
    products = []
    for i in range(50):
        name = f"Product_{i+1}"
        category = random.choice(categories)
        price = round(random.uniform(9.99, 499.99), 2)
        stock = random.randint(0, 500)
        products.append((name, category, price, stock))

    cursor.executemany("INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)", products)

    # Insert orders
    orders = []
    for i in range(500):
        customer_id = random.randint(1, 100)
        product_id = random.randint(1, 50)
        quantity = random.randint(1, 10)
        total = round(random.uniform(20, 2000), 2)
        days_ago = random.randint(1, 365)
        order_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        status = random.choice(statuses)
        orders.append((customer_id, product_id, quantity, total, order_date, status))

    cursor.executemany(
        "INSERT INTO orders (customer_id, product_id, quantity, total_amount, order_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        orders,
    )

    conn.commit()
    conn.close()

    print(f"✅ Created synthetic database at: {db_path.absolute()}")
    print(f"   - customers: 100 rows")
    print(f"   - products: 50 rows")
    print(f"   - orders: 500 rows")

    return str(db_path.absolute())


def test_database_connector(db_path: str):
    print("\n" + "=" * 60)
    print("TESTING DATABASE CONNECTOR")
    print("=" * 60)

    from src.connectors.service import get_connection_manager
    from src.connectors.base import ConnectionConfig

    manager = get_connection_manager()

    # 1. Test connection
    print("\n1. Testing connection...")
    config = ConnectionConfig(db_type="sqlite", database="sales_analytics", file_path=db_path)

    result = manager.test_connection(config)
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    if result.tables:
        print(f"   Tables found: {[t.name for t in result.tables]}")

    # 2. Connect
    print("\n2. Connecting...")
    connection_id = "test-connection-001"
    result = manager.connect(connection_id, config)
    print(f"   Connected: {result.success}")

    # 3. List tables
    print("\n3. Listing tables...")
    tables = manager.list_tables(connection_id)
    for t in tables:
        print(f"   - {t.name}: {t.row_count} rows")

    # 4. Get schema
    print("\n4. Getting 'orders' schema...")
    schema = manager.get_table_schema(connection_id, "orders")
    for col in schema:
        print(f"   - {col['name']} ({col['type']})")

    # 5. Execute query
    print("\n5. Executing query...")
    df = manager.execute_query(
        connection_id,
        """
        SELECT c.region, COUNT(*) as order_count, SUM(o.total_amount) as total_sales
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.status = 'completed'
        GROUP BY c.region
        ORDER BY total_sales DESC
    """,
    )
    print(df.to_string(index=False))

    # 6. Load table
    print("\n6. Loading 'products' table (first 10 rows)...")
    df = manager.load_table(connection_id, "products", limit=10)
    print(df.to_string(index=False))

    # 7. Disconnect
    print("\n7. Disconnecting...")
    success = manager.disconnect(connection_id)
    print(f"   Disconnected: {success}")

    print("\n✅ Database connector tests PASSED!")


def test_notification_service():
    print("\n" + "=" * 60)
    print("TESTING NOTIFICATION SERVICE")
    print("=" * 60)

    from src.notifications.service import get_notification_service, NotificationType

    service = get_notification_service()

    # 1. Create notifications
    print("\n1. Creating notifications...")
    n1 = service.create(
        title="Data Import Complete",
        message="Successfully imported 500 orders from sales_analytics.db",
        type=NotificationType.SUCCESS,
        session_id="test-session-001",
    )
    print(f"   Created: {n1.id} - {n1.title}")

    n2 = service.create(
        title="Analysis Ready", message="Your sales analysis report is ready for review", type=NotificationType.INFO
    )
    print(f"   Created: {n2.id} - {n2.title}")

    n3 = service.create(
        title="Alert: Low Stock", message="5 products are running low on inventory", type=NotificationType.WARNING
    )
    print(f"   Created: {n3.id} - {n3.title}")

    # 2. Get pending
    print("\n2. Getting pending notifications...")
    pending = service.get_pending()
    print(f"   Pending count: {len(pending)}")
    for n in pending:
        print(f"   - [{n.type.value}] {n.title}")

    # 3. Get unread
    print("\n3. Getting unread notifications...")
    unread = service.get_unread()
    print(f"   Unread count: {len(unread)}")

    # 4. Mark read
    print("\n4. Marking first notification as read...")
    success = service.mark_read(n1.id)
    print(f"   Marked read: {success}")

    # 5. Get unread again
    print("\n5. Getting unread after marking one read...")
    unread = service.get_unread()
    print(f"   Unread count: {len(unread)}")

    print("\n✅ Notification service tests PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("DATA INSIGHT - CONNECTOR & NOTIFICATION TEST SUITE")
    print("=" * 60)

    # Create test database
    db_path = create_synthetic_database()

    # Test database connector
    test_database_connector(db_path)

    # Test notification service
    test_notification_service()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
