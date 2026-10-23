from backend.app.ml.features import build_feature_matrix, make_dataset, transaction_from_payload


def test_generated_dataset_has_imbalanced_target_and_features():
    bundle = make_dataset(n_rows=600, seed=123)

    assert len(bundle.transactions) == 600
    assert bundle.target.sum() > 0
    assert bundle.target.mean() < 0.20
    assert "amount_to_customer_avg" in bundle.features.columns
    assert any(col.startswith("merchant_category_") for col in bundle.features.columns)


def test_payload_feature_matrix_reindexes_to_training_columns():
    bundle = make_dataset(n_rows=300, seed=99)
    payload_frame = transaction_from_payload(
        {
            "amount": 900,
            "merchant_category": "jewelry",
            "channel": "online",
            "customer_avg_amount": 80,
            "customer_std_amount": 30,
        }
    )

    matrix = build_feature_matrix(payload_frame, bundle.features.columns)

    assert list(matrix.columns) == list(bundle.features.columns)
    assert matrix.iloc[0]["amount"] == 900
    assert matrix.iloc[0]["merchant_category_jewelry"] == 1

