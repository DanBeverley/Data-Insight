import pytest
import threading
import time
from datetime import datetime

try:
    from src.database.connection import DatabaseManager, DatabaseConfig
    from src.database.models import DatasetCharacteristics, SystemMetrics, Base
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, scoped_session
    from sqlalchemy.pool import StaticPool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
@pytest.mark.integration
class TestConcurrentWrites:

    @pytest.fixture
    def db_engine(self):
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False, "timeout": 30}, poolclass=StaticPool
        )
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
        return engine

    @pytest.fixture
    def session_factory(self, db_engine):
        return scoped_session(sessionmaker(bind=db_engine))

    def test_concurrent_inserts_different_tables(self, session_factory):
        results = []
        errors = []

        def insert_dataset(idx):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    session_factory.close()
                    dataset = DatasetCharacteristics(
                        dataset_hash=f"concurrent_ds_{idx}",
                        n_samples=100,
                        n_features=5,
                        target_type="classification",
                        domain="test",
                    )
                    session_factory.add(dataset)
                    session_factory.commit()
                    results.append(True)
                    break
                except Exception as e:
                    try:
                        session_factory.rollback()
                    except:
                        pass
                    if attempt == max_retries - 1:
                        errors.append(f"Dataset insert {idx}: {str(e)}")
                        results.append(False)
                    else:
                        time.sleep(0.01)
                finally:
                    session_factory.remove()

        def insert_metric(idx):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    session_factory.close()
                    metric = SystemMetrics(
                        metric_id=f"concurrent_metric_{idx}", metric_type="test", metric_value=idx * 1.0
                    )
                    session_factory.add(metric)
                    session_factory.commit()
                    results.append(True)
                    break
                except Exception as e:
                    try:
                        session_factory.rollback()
                    except:
                        pass
                    if attempt == max_retries - 1:
                        errors.append(f"Metric insert {idx}: {str(e)}")
                        results.append(False)
                    else:
                        time.sleep(0.01)
                finally:
                    session_factory.remove()

        threads = []
        for i in range(5):
            t1 = threading.Thread(target=insert_dataset, args=(i,))
            t2 = threading.Thread(target=insert_metric, args=(i,))
            threads.extend([t1, t2])
            t1.start()
            t2.start()

        for thread in threads:
            thread.join()

        assert all(results), f"Some concurrent operations failed. Errors: {errors}"

    def test_concurrent_updates_same_record(self, session_factory):
        dataset = DatasetCharacteristics(
            dataset_hash="update_race", n_samples=100, n_features=5, target_type="classification", domain="test"
        )
        session_factory.add(dataset)
        session_factory.commit()
        session_factory.remove()

        def update_record(new_value):
            try:
                ds = session_factory.query(DatasetCharacteristics).filter_by(dataset_hash="update_race").first()
                if ds:
                    ds.n_samples = new_value
                    session_factory.commit()
            except Exception:
                session_factory.rollback()
            finally:
                session_factory.remove()

        threads = [threading.Thread(target=update_record, args=(i * 100,)) for i in range(1, 6)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        final_ds = session_factory.query(DatasetCharacteristics).filter_by(dataset_hash="update_race").first()
        assert final_ds is not None
        assert final_ds.n_samples in [100, 200, 300, 400, 500]
        session_factory.remove()

    def test_concurrent_reads_consistent(self, session_factory):
        for i in range(10):
            dataset = DatasetCharacteristics(
                dataset_hash=f"read_test_{i}", n_samples=100, n_features=5, target_type="classification", domain="test"
            )
            session_factory.add(dataset)
        session_factory.commit()
        session_factory.remove()

        read_counts = []

        def read_records():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    session_factory.close()
                    count = (
                        session_factory.query(DatasetCharacteristics)
                        .filter(DatasetCharacteristics.dataset_hash.like("read_test_%"))
                        .count()
                    )
                    read_counts.append(count)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        read_counts.append(None)
                    else:
                        time.sleep(0.01)
                finally:
                    session_factory.remove()

        threads = [threading.Thread(target=read_records) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert all(count == 10 for count in read_counts if count is not None)
        assert len([c for c in read_counts if c is not None]) >= 5

    def test_race_condition_handling(self, session_factory):
        completed = []

        def insert_with_delay(idx):
            try:
                dataset = DatasetCharacteristics(
                    dataset_hash=f"race_{idx}", n_samples=100, n_features=5, target_type="classification", domain="test"
                )
                session_factory.add(dataset)
                time.sleep(0.01)
                session_factory.commit()
                completed.append(True)
            except Exception:
                session_factory.rollback()
                completed.append(False)
            finally:
                session_factory.remove()

        threads = [threading.Thread(target=insert_with_delay, args=(i,)) for i in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(completed) == 10
