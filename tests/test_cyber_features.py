"""Comprehensive tests for cyber security feature engineering."""
import numpy as np
import pandas as pd
import pytest

from uais.features.cyber_features import build_cyber_feature_table


class TestBuildCyberFeatureTable:
    """Test cyber feature engineering pipeline."""

    def test_basic_feature_table(self):
        """Test basic feature table construction."""
        df = pd.DataFrame({
            'dur': [1.0, 2.0, 3.0, 4.0],
            'sbytes': [100, 200, 300, 400],
            'dbytes': [50, 100, 150, 200],
            'proto': ['tcp', 'udp', 'tcp', 'icmp'],
            'state': ['FIN', 'INT', 'FIN', 'CON'],
            'label': [0, 0, 1, 0]
        })

        result = build_cyber_feature_table(df, target_column='label')

        # Check target column
        assert 'label' in result.columns
        assert result['label'].dtype == int

        # Check rate features created
        assert 'bytes_per_sec_src' in result.columns
        assert 'bytes_per_sec_dst' in result.columns

        # Verify calculations
        expected_src_rate = [100/1.0, 200/2.0, 300/3.0, 400/4.0]
        np.testing.assert_array_almost_equal(
            result['bytes_per_sec_src'].values,
            expected_src_rate
        )

    def test_categorical_encoding(self):
        """Test categorical column encoding."""
        df = pd.DataFrame({
            'proto': ['tcp', 'udp', 'tcp', 'icmp', 'tcp'],
            'state': ['FIN', 'INT', 'FIN', 'CON', 'INT'],
            'score': [1.0, 2.0, 3.0, 4.0, 5.0],
            'label': [0, 0, 1, 0, 1]
        })

        result = build_cyber_feature_table(df, freq_encode_cats=True)

        # Check categorical columns are encoded
        assert pd.api.types.is_numeric_dtype(result['proto'])
        assert pd.api.types.is_numeric_dtype(result['state'])

        # Check frequency encoding columns exist
        assert 'proto_freq' in result.columns
        assert 'state_freq' in result.columns

        # Verify frequency encoding correctness
        # 'tcp' appears 3 times out of 5 = 0.6
        tcp_indices = df['proto'] == 'tcp'
        assert all(result.loc[tcp_indices, 'proto_freq'] == 0.6)

    def test_drop_columns(self):
        """Test dropping specified columns."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'score': [1.0, 2.0, 3.0],
            'label': [0, 1, 0]
        })

        result = build_cyber_feature_table(
            df,
            drop_columns=['id', 'timestamp']
        )

        # Check columns are dropped
        assert 'id' not in result.columns
        assert 'timestamp' not in result.columns
        assert 'score' in result.columns
        assert 'label' in result.columns

    def test_missing_target_raises_error(self):
        """Test that missing target column raises KeyError."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

        with pytest.raises(KeyError, match="Target column.*not found"):
            build_cyber_feature_table(df, target_column='label')

    def test_handles_string_labels(self):
        """Test conversion of string labels to integers."""
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0],
            'label': ['normal', 'attack', 'normal']
        })

        result = build_cyber_feature_table(df)

        # Label should be encoded as integers
        assert result['label'].dtype == int
        assert len(result) == 3

    def test_handles_inf_and_nan_in_target(self):
        """Test handling of infinite and NaN values in target."""
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'label': [0, 1, np.inf, -np.inf, np.nan]
        })

        result = build_cyber_feature_table(df)

        # Only first two rows should remain
        assert len(result) == 2
        assert all(np.isfinite(result['label']))

    def test_handles_zero_duration(self):
        """Test handling of zero duration in rate calculations."""
        df = pd.DataFrame({
            'dur': [0.0, 1.0, 2.0],
            'sbytes': [100, 200, 300],
            'dbytes': [50, 100, 150],
            'label': [0, 0, 1]
        })

        result = build_cyber_feature_table(df)

        # Zero duration should result in NaN for rate features
        assert pd.isna(result['bytes_per_sec_src'].iloc[0])
        assert not pd.isna(result['bytes_per_sec_src'].iloc[1])

    def test_without_rate_features(self):
        """Test feature table without duration/bytes columns."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'proto': ['tcp', 'udp', 'tcp'],
            'label': [0, 1, 0]
        })

        result = build_cyber_feature_table(df)

        # Rate features should not exist
        assert 'bytes_per_sec_src' not in result.columns
        assert 'bytes_per_sec_dst' not in result.columns

        # Basic encoding should still work
        assert 'label' in result.columns
        assert pd.api.types.is_numeric_dtype(result['proto'])

    def test_frequency_encoding_disabled(self):
        """Test disabling frequency encoding."""
        df = pd.DataFrame({
            'proto': ['tcp', 'udp', 'tcp'],
            'score': [1.0, 2.0, 3.0],
            'label': [0, 1, 0]
        })

        result = build_cyber_feature_table(df, freq_encode_cats=False)

        # Frequency columns should not exist
        assert 'proto_freq' not in result.columns

        # Basic label encoding should still happen
        assert pd.api.types.is_numeric_dtype(result['proto'])

    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'label': [0, 1, 0]
        })

        result = build_cyber_feature_table(df)

        # All should be numeric after encoding
        assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)


class TestCyberFeaturesEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['feature', 'label'])

        result = build_cyber_feature_table(df)
        assert len(result) == 0

    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            'feature': [1.0],
            'proto': ['tcp'],
            'label': [0]
        })

        result = build_cyber_feature_table(df)
        assert len(result) == 1

    def test_single_categorical_value(self):
        """Test handling of categorical column with single unique value."""
        df = pd.DataFrame({
            'proto': ['tcp', 'tcp', 'tcp'],
            'score': [1.0, 2.0, 3.0],
            'label': [0, 1, 0]
        })

        result = build_cyber_feature_table(df, freq_encode_cats=True)

        # All values should have same frequency (1.0)
        assert all(result['proto_freq'] == 1.0)

    def test_large_number_of_categories(self):
        """Test handling of high-cardinality categorical columns."""
        n = 1000
        df = pd.DataFrame({
            'id': range(n),  # Unique IDs
            'proto': [f'proto_{i}' for i in range(n)],
            'score': np.random.randn(n),
            'label': np.random.randint(0, 2, n)
        })

        result = build_cyber_feature_table(df)

        # Should handle gracefully
        assert len(result) == n
        assert 'proto' in result.columns


@pytest.fixture
def sample_unsw_data():
    """Fixture providing sample UNSW-NB15 style data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'dur': np.random.exponential(2.0, n),
        'sbytes': np.random.randint(0, 10000, n),
        'dbytes': np.random.randint(0, 10000, n),
        'sttl': np.random.randint(0, 256, n),
        'dttl': np.random.randint(0, 256, n),
        'proto': np.random.choice(['tcp', 'udp', 'icmp'], n),
        'state': np.random.choice(['FIN', 'INT', 'CON', 'REQ'], n),
        'service': np.random.choice(['http', 'ftp', 'ssh', 'dns'], n),
        'label': np.random.binomial(1, 0.2, n)
    })


def test_cyber_features_integration(sample_unsw_data):
    """Integration test for full cyber feature pipeline."""
    result = build_cyber_feature_table(sample_unsw_data, freq_encode_cats=True)

    # Check shape
    assert len(result) == len(sample_unsw_data)

    # Check all columns are numeric
    assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)

    # Check no NaN in target
    assert result['label'].notna().all()

    # Check rate features exist
    assert 'bytes_per_sec_src' in result.columns
    assert 'bytes_per_sec_dst' in result.columns

    # Check frequency encoding columns
    assert 'proto_freq' in result.columns
    assert 'state_freq' in result.columns
    assert 'service_freq' in result.columns

    # Verify frequency encoding is valid (between 0 and 1)
    assert (result['proto_freq'] >= 0).all() and (result['proto_freq'] <= 1).all()
