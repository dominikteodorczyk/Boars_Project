import pytest
import pandas as pd
from src.utils.parsers import parse_id, test_data_detector

@pytest.fixture
def sample_dataframe():
    """
    Fixture to create a sample DataFrame for testing.
    """
    data = {
        'timestamp': ['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00'],
        'user_id': ['user1', 'test_user', 'user2'],
        'lon': [1.0, 2.0, 3.0],
        'lat': [4.0, 5.0, 6.0]
    }
    return pd.DataFrame(data)


if __name__ == '__main__':
    pytest.main()