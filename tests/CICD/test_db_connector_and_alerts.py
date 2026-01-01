"""
Test suite for Database Connector and Schedule Alert features.
Creates a complex business database (e-commerce) for realistic testing.
"""

import asyncio
import sqlite3
import tempfile
import uuid
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectors.base import ConnectionConfig, TableInfo
from src.connectors.service import ConnectionManager, get_connection_manager
from src.connectors.sqlite import SQLiteConnector
from src.scheduler.models import Alert, AlertCondition, AlertStatus, AlertCreateRequest
from src.scheduler.service import AlertScheduler, get_alert_scheduler
from src.api_utils.session_management import session_data_manager


class ECommerceTestDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def create(self):
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        self._populate_data()
        self.conn.commit()
        return self

    def _create_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                phone TEXT,
                address TEXT,
                city TEXT,
                country TEXT,
                segment TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                lifetime_value REAL DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                unit_price REAL NOT NULL,
                cost_price REAL NOT NULL,
                stock_quantity INTEGER DEFAULT 0,
                reorder_level INTEGER DEFAULT 10,
                supplier_id INTEGER,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_number TEXT UNIQUE NOT NULL,
                customer_id INTEGER NOT NULL,
                order_date DATETIME NOT NULL,
                status TEXT NOT NULL,
                subtotal REAL NOT NULL,
                tax REAL DEFAULT 0,
                shipping_cost REAL DEFAULT 0,
                total REAL NOT NULL,
                payment_method TEXT,
                shipping_method TEXT,
                ship_date DATETIME,
                delivered_date DATETIME,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                discount REAL DEFAULT 0,
                line_total REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            
            CREATE TABLE IF NOT EXISTS inventory_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                reference_id TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            
            CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact_name TEXT,
                email TEXT,
                phone TEXT,
                address TEXT,
                country TEXT,
                lead_time_days INTEGER DEFAULT 7,
                payment_terms TEXT,
                is_active BOOLEAN DEFAULT 1
            );
            
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                revenue REAL DEFAULT 0,
                order_count INTEGER DEFAULT 0,
                avg_order_value REAL DEFAULT 0,
                new_customers INTEGER DEFAULT 0,
                conversion_rate REAL DEFAULT 0,
                cart_abandonment_rate REAL DEFAULT 0
            );
        """
        )

    def _populate_data(self):
        segments = ["Premium", "Standard", "Budget", "Enterprise", "Startup"]
        countries = ["USA", "UK", "Germany", "France", "Canada", "Australia", "Japan"]
        cities = ["New York", "London", "Berlin", "Paris", "Toronto", "Sydney", "Tokyo"]

        for i in range(500):
            self.conn.execute(
                """
                INSERT INTO customers (email, name, phone, address, city, country, segment, lifetime_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    f"customer{i}@email.com",
                    f"Customer {i}",
                    f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}",
                    f"{random.randint(100,999)} Main St",
                    random.choice(cities),
                    random.choice(countries),
                    random.choice(segments),
                    round(random.uniform(100, 10000), 2),
                ),
            )

        self.conn.execute(
            """
            INSERT INTO suppliers (name, contact_name, email, country, lead_time_days, payment_terms)
            VALUES 
            ('TechSupply Co', 'John Smith', 'john@techsupply.com', 'USA', 5, 'Net 30'),
            ('Global Parts Ltd', 'Jane Doe', 'jane@globalparts.com', 'UK', 7, 'Net 45'),
            ('Asian Electronics', 'Wei Chen', 'wei@asianelec.com', 'China', 14, 'Net 60'),
            ('EuroComponents', 'Hans Mueller', 'hans@eurocomp.de', 'Germany', 10, 'Net 30')
        """
        )

        categories = {
            "Electronics": ["Laptops", "Phones", "Tablets", "Accessories"],
            "Clothing": ["Men", "Women", "Kids", "Sports"],
            "Home & Garden": ["Furniture", "Decor", "Kitchen", "Outdoor"],
            "Office": ["Supplies", "Equipment", "Software", "Furniture"],
        }

        product_id = 0
        for category, subcats in categories.items():
            for subcat in subcats:
                for j in range(25):
                    product_id += 1
                    cost = round(random.uniform(10, 500), 2)
                    margin = random.uniform(1.2, 2.5)
                    self.conn.execute(
                        """
                        INSERT INTO products (sku, name, category, subcategory, unit_price, cost_price, 
                                            stock_quantity, reorder_level, supplier_id, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            f"SKU-{category[:3].upper()}-{subcat[:3].upper()}-{j:04d}",
                            f"{subcat} Product {j+1}",
                            category,
                            subcat,
                            round(cost * margin, 2),
                            cost,
                            random.randint(0, 500),
                            random.randint(10, 50),
                            random.randint(1, 4),
                            random.choice([1, 1, 1, 1, 0]),
                        ),
                    )

        statuses = ["pending", "processing", "shipped", "delivered", "cancelled", "refunded"]
        payment_methods = ["credit_card", "paypal", "bank_transfer", "crypto"]
        shipping_methods = ["standard", "express", "overnight", "pickup"]

        base_date = datetime.now() - timedelta(days=365)

        for i in range(2000):
            customer_id = random.randint(1, 500)
            order_date = base_date + timedelta(days=random.randint(0, 365))
            status = random.choice(statuses)

            num_items = random.randint(1, 5)
            subtotal = 0

            self.conn.execute(
                """
                INSERT INTO orders (order_number, customer_id, order_date, status, subtotal, 
                                   tax, shipping_cost, total, payment_method, shipping_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    f"ORD-{order_date.year}-{i+1:06d}",
                    customer_id,
                    order_date.isoformat(),
                    status,
                    0,
                    0,
                    0,
                    0,
                    random.choice(payment_methods),
                    random.choice(shipping_methods),
                ),
            )

            order_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            for _ in range(num_items):
                product_id = random.randint(1, 400)
                quantity = random.randint(1, 10)

                cursor = self.conn.execute("SELECT unit_price FROM products WHERE id = ?", (product_id,))
                row = cursor.fetchone()
                unit_price = row[0] if row else 50.0

                discount = round(random.uniform(0, 0.15) * unit_price * quantity, 2)
                line_total = round(unit_price * quantity - discount, 2)
                subtotal += line_total

                self.conn.execute(
                    """
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount, line_total)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (order_id, product_id, quantity, unit_price, discount, line_total),
                )

            tax = round(subtotal * 0.08, 2)
            shipping = round(random.uniform(5, 25), 2) if subtotal < 100 else 0
            total = round(subtotal + tax + shipping, 2)

            self.conn.execute(
                """
                UPDATE orders SET subtotal = ?, tax = ?, shipping_cost = ?, total = ? WHERE id = ?
            """,
                (subtotal, tax, shipping, total, order_id),
            )

        for i in range(365):
            date = (base_date + timedelta(days=i)).date().isoformat()
            revenue = round(random.uniform(5000, 50000), 2)
            orders = random.randint(20, 200)

            self.conn.execute(
                """
                INSERT INTO daily_metrics (date, revenue, order_count, avg_order_value, 
                                          new_customers, conversion_rate, cart_abandonment_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date,
                    revenue,
                    orders,
                    round(revenue / orders, 2),
                    random.randint(1, 30),
                    round(random.uniform(0.01, 0.08), 4),
                    round(random.uniform(0.60, 0.80), 4),
                ),
            )

    def close(self):
        if self.conn:
            self.conn.close()

    def export_to_csv(self, table_name: str, output_path: str):
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        df.to_csv(output_path, index=False)
        return output_path


class TestDatabaseConnector:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ecommerce.db")
        self.test_db = None
        self.manager = None

    def setup(self):
        print("\n" + "=" * 60)
        print("SETTING UP TEST DATABASE")
        print("=" * 60)

        self.test_db = ECommerceTestDatabase(self.db_path).create()
        self.manager = ConnectionManager(db_path=os.path.join(self.temp_dir, "test_connections.db"))

        print(f"Created test database at: {self.db_path}")
        return self

    def test_connection_config(self):
        print("\n" + "-" * 40)
        print("TEST: Connection Configuration")
        print("-" * 40)

        config = ConnectionConfig(db_type="sqlite", database=self.db_path, file_path=self.db_path)

        assert config.db_type == "sqlite"
        assert config.file_path == self.db_path
        print("✓ ConnectionConfig created successfully")

        return config

    def test_sqlite_connector(self):
        print("\n" + "-" * 40)
        print("TEST: SQLite Connector")
        print("-" * 40)

        config = ConnectionConfig(db_type="sqlite", database=self.db_path, file_path=self.db_path)

        connector = SQLiteConnector(config)
        result = connector.test_connection()

        assert result.success, f"Connection failed: {result.error}"
        print(f"✓ Connection test passed: {result.message}")
        print(f"  Tables found: {len(result.tables)}")

        for table in result.tables:
            print(f"    - {table.name}: {table.row_count} rows")

        result = connector.connect()
        assert result.success
        print("✓ Connect successful")

        tables = connector.list_tables()
        expected_tables = [
            "customers",
            "products",
            "orders",
            "order_items",
            "inventory_transactions",
            "suppliers",
            "daily_metrics",
        ]

        for expected in expected_tables:
            assert any(t.name == expected for t in tables), f"Missing table: {expected}"
        print(f"✓ All {len(expected_tables)} expected tables found")

        schema = connector.get_table_schema("orders")
        column_names = [col["name"] for col in schema]
        assert "customer_id" in column_names
        assert "total" in column_names
        print(f"✓ Schema retrieved for 'orders': {len(schema)} columns")

        df = connector.execute_query(
            """
            SELECT 
                c.segment,
                COUNT(o.id) as order_count,
                SUM(o.total) as total_revenue,
                AVG(o.total) as avg_order_value
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE o.status = 'delivered'
            GROUP BY c.segment
            ORDER BY total_revenue DESC
        """
        )

        assert len(df) > 0
        assert "segment" in df.columns
        assert "total_revenue" in df.columns
        print(f"✓ Complex aggregation query executed: {len(df)} rows")
        print(f"  Top segment: {df.iloc[0]['segment']} - ${df.iloc[0]['total_revenue']:,.2f}")

        df_products = connector.load_table("products", limit=100)
        assert len(df_products) == 100
        print(f"✓ Table load with limit: {len(df_products)} rows")

        connector.disconnect()
        print("✓ Disconnect successful")

        return True

    def test_connection_manager(self):
        print("\n" + "-" * 40)
        print("TEST: Connection Manager")
        print("-" * 40)

        config = ConnectionConfig(db_type="sqlite", database=self.db_path, file_path=self.db_path)

        connection_id = str(uuid.uuid4())
        result = self.manager.test_connection(config)
        assert result.success
        print(f"✓ Manager test_connection passed")

        result = self.manager.connect(connection_id, config)
        assert result.success
        print(f"✓ Manager connect passed (ID: {connection_id[:8]}...)")

        tables = self.manager.list_tables(connection_id)
        assert len(tables) >= 7
        print(f"✓ Manager list_tables: {len(tables)} tables")

        schema = self.manager.get_table_schema(connection_id, "customers")
        assert len(schema) > 0
        print(f"✓ Manager get_table_schema: {len(schema)} columns")

        df = self.manager.execute_query(
            connection_id,
            """
            SELECT category, SUM(stock_quantity) as total_stock
            FROM products
            GROUP BY category
        """,
        )
        assert len(df) > 0
        print(f"✓ Manager execute_query: {len(df)} rows")

        df_loaded = self.manager.load_table(connection_id, "daily_metrics")
        assert len(df_loaded) > 0
        print(f"✓ Manager load_table (daily_metrics): {len(df_loaded)} rows")

        self.manager.save_connection(connection_id, "Test E-Commerce DB", config)
        saved = self.manager.get_saved_connections()
        assert any(c["id"] == connection_id for c in saved)
        print(f"✓ Connection saved and retrieved")

        loaded_config = self.manager.load_saved_connection(connection_id)
        assert loaded_config is not None
        assert loaded_config.db_type == "sqlite"
        print(f"✓ Saved connection config loaded")

        success = self.manager.disconnect(connection_id)
        assert success
        print(f"✓ Manager disconnect passed")

        return True

    def cleanup(self):
        if self.test_db:
            self.test_db.close()

        import shutil

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


class TestAlertScheduler:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ecommerce.db")
        self.alerts_db_path = os.path.join(self.temp_dir, "test_alerts.db")
        self.test_db = None
        self.scheduler = None
        self.session_id = str(uuid.uuid4())

    def setup(self):
        print("\n" + "=" * 60)
        print("SETTING UP ALERT SCHEDULER TEST")
        print("=" * 60)

        self.test_db = ECommerceTestDatabase(self.db_path).create()
        self.scheduler = AlertScheduler(db_path=self.alerts_db_path)

        csv_path = os.path.join(self.temp_dir, "daily_metrics.csv")
        self.test_db.export_to_csv("daily_metrics", csv_path)

        session_data = session_data_manager.create_session(self.session_id)
        session_data["dataset_path"] = csv_path
        session_data["dataset_name"] = "daily_metrics"

        print(f"Created test session: {self.session_id[:8]}...")
        print(f"Dataset exported to: {csv_path}")
        return self

    def test_alert_models(self):
        print("\n" + "-" * 40)
        print("TEST: Alert Models")
        print("-" * 40)

        for condition in AlertCondition:
            print(f"  ✓ {condition.name}: {condition.value}")

        for status in AlertStatus:
            print(f"  ✓ {status.name}: {status.value}")

        alert = Alert(
            name="Test Revenue Alert",
            session_id=self.session_id,
            metric_query="df['revenue'].mean()",
            metric_name="Average Daily Revenue",
            condition=AlertCondition.LESS_THAN,
            threshold=10000.0,
            cron_expression="0 9 * * *",
            notification_type="email",
            notification_target="admin@company.com",
        )

        assert alert.id is not None
        assert alert.status == AlertStatus.ACTIVE
        print(f"✓ Alert model created: {alert.name}")
        print(f"  ID: {alert.id[:8]}...")
        print(f"  Metric: {alert.metric_name}")
        print(f"  Condition: {alert.condition} {alert.threshold}")

        return True

    def test_alert_crud(self):
        print("\n" + "-" * 40)
        print("TEST: Alert CRUD Operations")
        print("-" * 40)

        alert1 = Alert(
            name="Low Revenue Alert",
            session_id=self.session_id,
            metric_query="df['revenue'].mean()",
            metric_name="Average Revenue",
            condition=AlertCondition.LESS_THAN,
            threshold=15000.0,
            notification_type="email",
            notification_target="finance@company.com",
        )

        alert2 = Alert(
            name="High Cart Abandonment Alert",
            session_id=self.session_id,
            metric_query="df['cart_abandonment_rate'].mean() * 100",
            metric_name="Cart Abandonment %",
            condition=AlertCondition.GREATER_THAN,
            threshold=75.0,
            notification_type="slack",
            notification_target="#ecommerce-alerts",
        )

        alert3 = Alert(
            name="Order Volume Alert",
            session_id=self.session_id,
            metric_query="df['order_count'].sum()",
            metric_name="Total Orders",
            condition=AlertCondition.LESS_THAN,
            threshold=50000,
            notification_type="webhook",
            notification_target="https://api.company.com/alerts",
        )

        created1 = self.scheduler.create_alert(alert1)
        created2 = self.scheduler.create_alert(alert2)
        created3 = self.scheduler.create_alert(alert3)

        print(f"✓ Created 3 alerts")

        retrieved = self.scheduler.get_alert(created1.id)
        assert retrieved is not None
        assert retrieved.name == "Low Revenue Alert"
        print(f"✓ Retrieved alert: {retrieved.name}")

        all_alerts = self.scheduler.get_all_alerts()
        assert len(all_alerts) >= 3
        print(f"✓ Get all alerts: {len(all_alerts)} alerts")

        session_alerts = self.scheduler.get_alerts_by_session(self.session_id)
        assert len(session_alerts) >= 3
        print(f"✓ Get session alerts: {len(session_alerts)} alerts")

        updated = self.scheduler.update_alert(created1.id, {"threshold": 20000.0, "name": "Updated Revenue Alert"})
        assert updated is not None
        assert updated.threshold == 20000.0
        print(f"✓ Updated alert threshold: {updated.threshold}")

        success = self.scheduler.delete_alert(created3.id)
        assert success
        remaining = self.scheduler.get_all_alerts()
        assert not any(a.id == created3.id for a in remaining)
        print(f"✓ Deleted alert successfully")

        return True

    def test_alert_check(self):
        print("\n" + "-" * 40)
        print("TEST: Alert Check Execution")
        print("-" * 40)

        alert = Alert(
            name="Revenue Check Alert",
            session_id=self.session_id,
            metric_query="df['revenue'].mean()",
            metric_name="Average Revenue",
            condition=AlertCondition.GREATER_THAN,
            threshold=1000.0,
            notification_type="log",
            notification_target="test",
        )

        created = self.scheduler.create_alert(alert)

        async def run_check():
            result = await self.scheduler.check_alert_now(created.id)
            return result

        result = asyncio.get_event_loop().run_until_complete(run_check())

        print(f"✓ Alert check executed")
        print(f"  Current Value: {result.current_value:,.2f}")
        print(f"  Threshold: {result.threshold:,.2f}")
        print(f"  Condition: {result.condition}")
        print(f"  Triggered: {result.triggered}")
        print(f"  Message: {result.message}")

        history = self.scheduler.get_alert_history(created.id)
        assert len(history) >= 1
        print(f"✓ Alert history recorded: {len(history)} entries")

        return True

    def test_condition_evaluation(self):
        print("\n" + "-" * 40)
        print("TEST: Condition Evaluation")
        print("-" * 40)

        test_cases = [
            (50, "lt", 100, True),
            (150, "lt", 100, False),
            (150, "gt", 100, True),
            (50, "gt", 100, False),
            (100, "eq", 100, True),
            (99, "eq", 100, False),
            (99, "ne", 100, True),
            (100, "ne", 100, False),
            (100, "lte", 100, True),
            (99, "lte", 100, True),
            (101, "lte", 100, False),
            (100, "gte", 100, True),
            (101, "gte", 100, True),
            (99, "gte", 100, False),
        ]

        for value, condition, threshold, expected in test_cases:
            result = self.scheduler._evaluate_condition(value, condition, threshold)
            status = "✓" if result == expected else "✗"
            print(f"  {status} {value} {condition} {threshold} = {result}")
            assert result == expected, f"Failed: {value} {condition} {threshold}"

        print(f"✓ All {len(test_cases)} condition evaluations passed")
        return True

    def test_pause_resume(self):
        print("\n" + "-" * 40)
        print("TEST: Alert Pause/Resume")
        print("-" * 40)

        alert = Alert(
            name="Pauseable Alert",
            session_id=self.session_id,
            metric_query="df['revenue'].sum()",
            metric_name="Total Revenue",
            condition=AlertCondition.GREATER_THAN,
            threshold=0,
            notification_type="log",
            notification_target="test",
        )

        created = self.scheduler.create_alert(alert)
        assert created.status == AlertStatus.ACTIVE.value
        print(f"✓ Alert created with status: {created.status}")

        paused = self.scheduler.update_alert(created.id, {"status": "paused"})
        assert paused.status == "paused"
        print(f"✓ Alert paused: {paused.status}")

        resumed = self.scheduler.update_alert(created.id, {"status": "active"})
        assert resumed.status == "active"
        print(f"✓ Alert resumed: {resumed.status}")

        return True

    def cleanup(self):
        if self.test_db:
            self.test_db.close()

        import shutil

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


def run_all_tests():
    print("\n" + "=" * 60)
    print("DATABASE CONNECTOR & ALERT SCHEDULER TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    results = {}

    print("\n\n" + "#" * 60)
    print("# PART 1: DATABASE CONNECTOR TESTS")
    print("#" * 60)

    connector_tests = TestDatabaseConnector()
    try:
        connector_tests.setup()

        connector_tests.test_connection_config()
        results["connection_config"] = "PASSED"

        connector_tests.test_sqlite_connector()
        results["sqlite_connector"] = "PASSED"

        connector_tests.test_connection_manager()
        results["connection_manager"] = "PASSED"

    except Exception as e:
        print(f"\n✗ CONNECTOR TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["connector_tests"] = f"FAILED: {e}"
    finally:
        connector_tests.cleanup()

    print("\n\n" + "#" * 60)
    print("# PART 2: ALERT SCHEDULER TESTS")
    print("#" * 60)

    alert_tests = TestAlertScheduler()
    try:
        alert_tests.setup()

        alert_tests.test_alert_models()
        results["alert_models"] = "PASSED"

        alert_tests.test_alert_crud()
        results["alert_crud"] = "PASSED"

        alert_tests.test_condition_evaluation()
        results["condition_evaluation"] = "PASSED"

        alert_tests.test_alert_check()
        results["alert_check"] = "PASSED"

        alert_tests.test_pause_resume()
        results["pause_resume"] = "PASSED"

    except Exception as e:
        print(f"\n✗ ALERT TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["alert_tests"] = f"FAILED: {e}"
    finally:
        alert_tests.cleanup()

    print("\n\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = sum(1 for v in results.values() if "FAILED" in v)

    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {test_name}: {result}")

    print("-" * 60)
    print(f"TOTAL: {passed} passed, {failed} failed")
    print(f"Completed at: {datetime.now().isoformat()}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
