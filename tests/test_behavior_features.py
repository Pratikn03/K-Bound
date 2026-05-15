"""Comprehensive tests for behavioral analytics feature engineering."""
import numpy as np
import pandas as pd
import pytest

from uais.features.behavior_features import build_behavior_feature_table


class TestBuildBehaviorFeatureTable:
    """Test behavior feature engineering pipeline."""

    def test_basic_feature_table(self):
        """Test basic feature table construction."""
        df = pd.DataFrame({
            'user': [1, 1, 2, 2],
            'date': ['2023-01-01 10:00', '2023-01-01 11:00',
                     '2023-01-01 12:00', '2023-01-01 14:00'],
            'PageValues': [1.5, 2.0, 3.0, 1.0],
            'Revenue': [False, True, False, True]
        })

        result = build_behavior_feature_table(df, target_column='Revenue')

        # Check target column
        assert 'Revenue' in result.columns
        assert result['Revenue'].dtype == int
        assert set(result['Revenue'].unique()) <= {0, 1}

        # Check temporal features created
        assert 'hour' in result.columns
        assert 'dayofweek' in result.columns

        # Check user features created
        assert 'events_per_user' in result.columns
        assert 'secs_since_last' in result.columns

    def test_temporal_feature_extraction(self):
        """Test extraction of temporal features."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01 10:30', '2023-01-02 14:45',
                                   '2023-01-03 18:15']),
            'feature': [1, 2, 3],
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df)

        # Verify hour extraction
        assert result['hour'].tolist() == [10, 14, 18]

        # Verify dayofweek (Sunday=6, Monday=0, ...)
        assert all(0 <= dow <= 6 for dow in result['dayofweek'])

    def test_user_sequence_features(self):
        """Test per-user sequence feature calculation."""
        df = pd.DataFrame({
            'user': [1, 1, 1, 2, 2],
            'date': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:05',
                                   '2023-01-01 10:10', '2023-01-01 11:00',
                                   '2023-01-01 11:10']),
            'Revenue': [0, 0, 1, 1, 0]
        })

        result = build_behavior_feature_table(df)

        # Check events_per_user is cumulative count
        user1_events = result[result.index.isin([0, 1, 2])]['events_per_user'].tolist()
        assert user1_events == [0, 1, 2]

        user2_events = result[result.index.isin([3, 4])]['events_per_user'].tolist()
        assert user2_events == [0, 1]

        # Check secs_since_last
        assert pd.isna(result.iloc[0]['secs_since_last']) or result.iloc[0]['secs_since_last'] >= 0
        assert result.iloc[1]['secs_since_last'] == 300  # 5 minutes = 300 seconds
        assert result.iloc[2]['secs_since_last'] == 300  # 5 minutes

    def test_drop_columns(self):
        """Test dropping specified columns."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'timestamp_raw': ['a', 'b', 'c'],
            'feature': [1.0, 2.0, 3.0],
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(
            df,
            drop_columns=['id', 'timestamp_raw']
        )

        # Check columns are dropped
        assert 'id' not in result.columns
        assert 'timestamp_raw' not in result.columns
        assert 'feature' in result.columns

    def test_missing_target_raises_error(self):
        """Test that missing target column raises KeyError."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

        with pytest.raises(KeyError, match="Target column.*not found"):
            build_behavior_feature_table(df, target_column='Revenue')

    def test_categorical_encoding(self):
        """Test categorical column encoding."""
        df = pd.DataFrame({
            'VisitorType': ['New', 'Returning', 'New', 'Other'],
            'Month': ['Jan', 'Feb', 'Jan', 'Mar'],
            'Weekend': [True, False, True, False],
            'PageValues': [1.0, 2.0, 3.0, 4.0],
            'Revenue': [0, 1, 0, 1]
        })

        result = build_behavior_feature_table(df)

        # All columns should be numeric after encoding
        assert pd.api.types.is_numeric_dtype(result['VisitorType'])
        assert pd.api.types.is_numeric_dtype(result['Month'])
        assert pd.api.types.is_numeric_dtype(result['Weekend'])

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'numeric_col': [1.0, np.nan, 3.0, 4.0],
            'cat_col': ['a', None, 'b', 'c'],
            'bool_col': [True, False, None, True],
            'Revenue': [0, 1, 0, 1]
        })

        result = build_behavior_feature_table(df)

        # Missing numeric should be filled with 0
        assert not pd.isna(result['numeric_col']).any()

        # Missing categorical should be filled with 'missing' then encoded
        assert not pd.isna(result['cat_col']).any()

        # Check no NaN in final result
        assert not result.isna().any().any()

    def test_without_time_column(self):
        """Test feature table without time column."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df, time_column='nonexistent')

        # Temporal features should not exist
        assert 'hour' not in result.columns
        assert 'dayofweek' not in result.columns

        # Basic features should still work
        assert 'Revenue' in result.columns
        assert len(result) == 3

    def test_without_user_column(self):
        """Test feature table without user column."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'feature': [1, 2, 3],
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df, user_column='nonexistent')

        # User-specific features should not exist
        assert 'events_per_user' not in result.columns
        assert 'secs_since_last' not in result.columns

        # Temporal features should still exist
        assert 'hour' in result.columns
        assert 'dayofweek' in result.columns


class TestBehaviorFeaturesEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['feature', 'Revenue'])

        result = build_behavior_feature_table(df)
        assert len(result) == 0

    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            'feature': [1.0],
            'date': ['2023-01-01 10:00'],
            'Revenue': [1]
        })

        result = build_behavior_feature_table(df)
        assert len(result) == 1

    def test_single_user(self):
        """Test handling of single user."""
        df = pd.DataFrame({
            'user': [1, 1, 1],
            'date': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00',
                                   '2023-01-01 12:00']),
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df)

        # events_per_user should be 0, 1, 2
        assert result['events_per_user'].tolist() == [0, 1, 2]

    def test_invalid_dates(self):
        """Test handling of invalid date strings."""
        df = pd.DataFrame({
            'date': ['invalid', '2023-01-01', 'also invalid'],
            'feature': [1, 2, 3],
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df)

        # Should handle gracefully with NaT -> filled values
        assert 'hour' in result.columns
        # Invalid dates should be coerced to some default or NaN->filled

    def test_boolean_target(self):
        """Test handling of boolean target column."""
        df = pd.DataFrame({
            'feature': [1, 2, 3],
            'Revenue': [True, False, True]
        })

        result = build_behavior_feature_table(df)

        # Target should be converted to int (0/1)
        assert result['Revenue'].dtype == int
        assert set(result['Revenue'].unique()) == {0, 1}

    def test_mixed_type_columns(self):
        """Test handling of mixed type columns."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df)

        # All should be numeric after processing
        assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)

    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps for same user."""
        df = pd.DataFrame({
            'user': [1, 1, 1],
            'date': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:00',
                                   '2023-01-01 11:00']),
            'Revenue': [0, 1, 0]
        })

        result = build_behavior_feature_table(df)

        # Should handle gracefully
        assert len(result) == 3
        assert 'secs_since_last' in result.columns

        # Second event should have 0 seconds since last
        assert result.iloc[1]['secs_since_last'] == 0


@pytest.fixture
def sample_shoppers_data():
    """Fixture providing sample Online Shoppers Intention data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'user': np.random.randint(1, 20, n),
        'date': pd.date_range('2023-01-01', periods=n, freq='1h'),
        'Administrative': np.random.randint(0, 10, n),
        'Informational': np.random.randint(0, 10, n),
        'ProductRelated': np.random.randint(0, 100, n),
        'BounceRates': np.random.uniform(0, 0.2, n),
        'ExitRates': np.random.uniform(0, 0.2, n),
        'PageValues': np.random.uniform(0, 50, n),
        'VisitorType': np.random.choice(['New', 'Returning', 'Other'], n),
        'Weekend': np.random.choice([True, False], n),
        'Revenue': np.random.binomial(1, 0.15, n)
    })


def test_behavior_features_integration(sample_shoppers_data):
    """Integration test for full behavior feature pipeline."""
    result = build_behavior_feature_table(sample_shoppers_data)

    # Check shape
    assert len(result) == len(sample_shoppers_data)

    # Check all columns are numeric
    assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)

    # Check no NaN in result
    assert not result.isna().any().any()

    # Check target is binary
    assert set(result['Revenue'].unique()) <= {0, 1}

    # Check temporal features exist
    assert 'hour' in result.columns
    assert 'dayofweek' in result.columns

    # Check user features exist
    assert 'events_per_user' in result.columns
    assert 'secs_since_last' in result.columns

    # Verify temporal features are valid
    assert (result['hour'] >= 0).all() and (result['hour'] <= 23).all()
    assert (result['dayofweek'] >= 0).all() and (result['dayofweek'] <= 6).all()
