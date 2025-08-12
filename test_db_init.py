import sys
import os
from unittest.mock import patch, MagicMock

# Add the folder where db_init.py is
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot/routes/utilities"))

# Set environment variables required by db_init.py
os.environ["REDIS_PASSWORD"] = "testpassword"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_DB"] = "0"

@patch("db_init.redis.Redis")
def test_redis_connection(mock_redis_class):
    print("\nStarting test_redis_connection")

    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_redis_class.return_value = mock_client

    import db_init  
    print("\nImported db_init module")

    client = db_init.Redis(0)
    print(f"\nCreated Redis client: {client}")

    ping_result = client.ping()
    print(f"\nPing result: {ping_result}")

    assert ping_result is True
    print("\nPing assertion passed")

    mock_redis_class.assert_called_once()
    print("\nRedis class was called once as expected")

