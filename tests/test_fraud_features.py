"""Comprehensive tests for fraud feature engineering."""

import numpy as np
import pandas as pd
import pytest

from uais.features.fraud_features import add_basic_fraud_features, build_fraud_feature_table


class TestAddBasicFraudFeatures:
    """Test basic fraud feature engineering."""

    def test_basic_features_creditcard(self):
        """Test feature engineering on creditcard-style data."""
        df = pd.DataFrame(
            {
                "Time": [0, 100, 500, 1000],
                "Amount": [10.0, 50.0, 150.0, 1000.0],
                "V1": [1.0, 2.0, 3.0, 4.0],
                "Class": [0, 0, 1, 0],
            }
        )

        result = add_basic_fraud_features(df, time_column="Time", amount_column="Amount")

        # Check new columns exist
        assert "amount_log" in result.columns
        assert "time_hours" in result.columns
        assert "hour_of_day" in result.columns

        # Check amount_log correctness
        np.testing.assert_array_almost_equal(result["amount_log"].values, np.log1p([10.0, 50.0, 150.0, 1000.0]))

        # Check time conversion
        expected_hours = np.array([0, 100, 500, 1000]) / 3600.0
        np.testing.assert_array_almost_equal(result["time_hours"].values, expected_hours)

    def test_basic_features_paysim(self):
        """Test feature engineering on PaySim-style data."""
        df = pd.DataFrame({"step": [1, 2, 3, 24], "amount": [100.0, 200.0, 50.0, 5000.0], "isFraud": [0, 0, 1, 0]})

        result = add_basic_fraud_features(df, time_column="step", amount_column="amount")

        # Check features exist
        assert "amount_log" in result.columns
        assert "time_hours" in result.columns

        # Verify amount_log
        expected_log = np.log1p([100.0, 200.0, 50.0, 5000.0])
        np.testing.assert_array_almost_equal(result["amount_log"].values, expected_log)

    def test_handles_missing_time_column(self):
        """Test graceful fallback when time column is missing."""
        df = pd.DataFrame({"Amount": [10.0, 50.0], "Class": [0, 1]})

        result = add_basic_fraud_features(df, time_column="Time", amount_column="Amount")

        # Should have zero-filled time features
        assert "time_hours" in result.columns
        assert all(result["time_hours"] == 0)

    def test_missing_amount_column_raises_error(self):
        """Test that missing amount column raises KeyError."""
        df = pd.DataFrame({"Time": [0, 100], "Class": [0, 1]})

        with pytest.raises(KeyError, match="Amount column not found"):
            add_basic_fraud_features(df, amount_column="Amount")

    def test_copy_parameter(self):
        """Test that copy parameter controls DataFrame mutation."""
        df = pd.DataFrame({"Time": [0, 100], "Amount": [10.0, 50.0], "Class": [0, 1]})
        original_columns = set(df.columns)

        # With copy=True (default)
        result = add_basic_fraud_features(df, copy=True)
        assert set(df.columns) == original_columns  # Original unchanged
        assert "amount_log" in result.columns

        # With copy=False
        result = add_basic_fraud_features(df, copy=False)
        assert "amount_log" in df.columns  # Original mutated


class TestBuildFraudFeatureTable:
    """Test full fraud feature pipeline."""

    def test_full_pipeline_creditcard(self):
        """Test full feature pipeline on creditcard data."""
        df = pd.DataFrame(
            {
                "Time": [0, 100, 500, 1000, 2000],
                "Amount": [10.0, 50.0, 150.0, 1000.0, 25.0],
                "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "V2": [0.5, 1.5, 2.5, 3.5, 4.5],
                "Class": [0, 0, 1, 0, 1],
            }
        )

        result = build_fraud_feature_table(df, target_column="Class")

        # Check target column preserved
        assert "Class" in result.columns
        assert result["Class"].dtype == int

        # Check feature columns exist
        assert "amount_log" in result.columns
        assert "time_hours" in result.columns

        # Check original columns preserved
        assert "V1" in result.columns
        assert "V2" in result.columns

    def test_drop_original_time(self):
        """Test that Time column can be dropped."""
        df = pd.DataFrame({"Time": [0, 100, 500], "Amount": [10.0, 50.0, 150.0], "Class": [0, 0, 1]})

        result = build_fraud_feature_table(df, drop_original_time=True)

        assert "Time" not in result.columns
        assert "time_hours" in result.columns

    def test_entity_aggregation_features(self):
        """Test per-entity aggregation features."""
        df = pd.DataFrame(
            {
                "Time": [0, 100, 200, 300, 400],
                "Amount": [10.0, 20.0, 15.0, 30.0, 25.0],
                "user_id": [1, 1, 2, 1, 2],
                "Class": [0, 0, 0, 1, 0],
            }
        )

        result = build_fraud_feature_table(df, group_id_column="user_id")

        # Check entity features exist
        assert "tx_count_per_entity" in result.columns
        assert "amount_running_mean" in result.columns
        assert "amount_running_std" in result.columns
        assert "amount_zscore_entity" in result.columns

    def test_handles_mixed_target_columns(self):
        """Test handling of both Class and isFraud columns."""
        df = pd.DataFrame(
            {"Time": [0, 100, 200], "Amount": [10.0, 50.0, 150.0], "Class": [0, np.nan, 1], "isFraud": [0, 1, np.nan]}
        )

        result = build_fraud_feature_table(df, target_column="Class")

        # Should fill NaN in Class from isFraud
        assert result["Class"].notna().all()
        assert len(result) == 3  # All rows preserved

    def test_missing_target_raises_error(self):
        """Test that missing target column raises KeyError."""
        df = pd.DataFrame({"Time": [0, 100], "Amount": [10.0, 50.0]})

        with pytest.raises(KeyError, match="Target column"):
            build_fraud_feature_table(df, target_column="Class")


class TestFraudFeaturesEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["Time", "Amount", "Class"])

        result = add_basic_fraud_features(df)
        assert len(result) == 0
        assert "amount_log" in result.columns

    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({"Time": [0], "Amount": [100.0], "Class": [1]})

        result = build_fraud_feature_table(df)
        assert len(result) == 1
        assert result["amount_log"].iloc[0] == np.log1p(100.0)

    def test_zero_and_negative_amounts(self):
        """Test handling of zero and negative amounts."""
        df = pd.DataFrame({"Time": [0, 100, 200], "Amount": [0.0, -10.0, 50.0], "Class": [0, 1, 0]})

        result = add_basic_fraud_features(df)

        # log1p should handle zero and negatives gracefully
        assert np.isfinite(result["amount_log"]).all()

    def test_large_time_values(self):
        """Test handling of very large time values."""
        df = pd.DataFrame(
            {
                "Time": [0, 86400, 172800],  # 0, 1 day, 2 days in seconds
                "Amount": [100.0, 200.0, 300.0],
                "Class": [0, 0, 1],
            }
        )

        result = add_basic_fraud_features(df)

        # Check hour_of_day wraps correctly
        assert result["hour_of_day"].iloc[0] == 0
        assert 0 <= result["hour_of_day"].iloc[1] < 24
        assert 0 <= result["hour_of_day"].iloc[2] < 24


@pytest.fixture
def sample_fraud_data():
    """Fixture providing sample fraud data for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "Time": np.random.randint(0, 172800, n),
            "Amount": np.random.exponential(100, n),
            "V1": np.random.randn(n),
            "V2": np.random.randn(n),
            "Class": np.random.binomial(1, 0.1, n),
        }
    )


def test_fraud_features_integration(sample_fraud_data):
    """Integration test for full fraud feature pipeline."""
    result = build_fraud_feature_table(sample_fraud_data)

    # Check shape
    assert len(result) == len(sample_fraud_data)

    # Check no NaN in critical columns
    assert result["amount_log"].notna().all()
    assert result["Class"].notna().all()

    # Check data types
    assert result["Class"].dtype == int
    assert pd.api.types.is_numeric_dtype(result["amount_log"])
