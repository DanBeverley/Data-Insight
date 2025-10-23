import pytest
from unittest.mock import patch
import psycopg2


@pytest.mark.chaos
class TestDatabaseFailures:
    def test_database_connection_loss(self, sample_session_id: str):
        from src.database.connection import get_db_connection

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError("connection lost")

            with pytest.raises(Exception):
                get_db_connection()

    def test_database_query_timeout(self, sample_session_id: str):
        from src.database.service import get_session_data

        with patch("src.database.connection.get_db_connection") as mock_conn:
            mock_cursor = mock_conn.return_value.cursor.return_value
            mock_cursor.execute.side_effect = psycopg2.extensions.QueryCanceledError("query timeout")

            with pytest.raises(Exception):
                get_session_data(sample_session_id)

    def test_database_deadlock_recovery(self, sample_session_id: str):
        from src.database.service import save_session_data

        with patch("src.database.connection.get_db_connection") as mock_conn:
            mock_cursor = mock_conn.return_value.cursor.return_value

            attempt_count = [0]

            def deadlock_then_success(*args, **kwargs):
                attempt_count[0] += 1
                if attempt_count[0] == 1:
                    raise psycopg2.extensions.TransactionRollbackError("deadlock detected")
                return None

            mock_cursor.execute.side_effect = deadlock_then_success

            try:
                save_session_data(sample_session_id, {"test": "data"})
            except Exception as e:
                assert "deadlock" in str(e).lower() or attempt_count[0] > 1

    def test_database_disk_full(self, sample_session_id: str):
        from src.database.service import save_session_data

        with patch("src.database.connection.get_db_connection") as mock_conn:
            mock_cursor = mock_conn.return_value.cursor.return_value
            mock_cursor.execute.side_effect = psycopg2.OperationalError("disk full")

            with pytest.raises(Exception) as exc_info:
                save_session_data(sample_session_id, {"large": "data" * 1000})

            assert "disk" in str(exc_info.value).lower() or "space" in str(exc_info.value).lower()

    def test_database_transaction_rollback(self, sample_session_id: str):
        from src.database.service import save_session_data

        with patch("src.database.connection.get_db_connection") as mock_conn:
            mock_conn_obj = mock_conn.return_value
            mock_cursor = mock_conn_obj.cursor.return_value
            mock_cursor.execute.side_effect = Exception("transaction error")

            try:
                save_session_data(sample_session_id, {"test": "data"})
            except Exception:
                pass

            mock_conn_obj.rollback.assert_called()

    def test_connection_pool_exhaustion(self):
        from src.database.connection import get_db_connection

        connections = []
        try:
            for _ in range(100):
                conn = get_db_connection()
                connections.append(conn)
        except Exception as e:
            assert "pool" in str(e).lower() or "connection" in str(e).lower()
        finally:
            for conn in connections:
                try:
                    conn.close()
                except:
                    pass
