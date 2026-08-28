import numpy as np
import pandas as pd

from open_the_valve.mlops.drift import build_drift_report


def _synthetic_panel(mean_shift: float, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "log_player_count": rng.normal(mean_shift, 1, n),
            "depth_pct": rng.integers(0, 100, n),
        }
    )


def test_build_drift_report_flags_an_obvious_shift(tmp_path):
    reference = _synthetic_panel(mean_shift=0.0)
    current = _synthetic_panel(mean_shift=5.0)  # obvious shift
    reference_cate = np.random.default_rng(0).normal(0, 0.5, len(reference))
    current_cate = np.random.default_rng(1).normal(5, 0.5, len(current))  # obvious shift
    output_path = tmp_path / "drift_report.html"

    dataset_drift_share, cate_drift_detected = build_drift_report(
        reference, current, reference_cate, current_cate, str(output_path)
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert dataset_drift_share > 0
    assert cate_drift_detected


def test_build_drift_report_no_shift(tmp_path):
    reference = _synthetic_panel(mean_shift=0.0)
    current = _synthetic_panel(mean_shift=0.0)
    reference_cate = np.random.default_rng(0).normal(0, 0.5, len(reference))
    current_cate = np.random.default_rng(0).normal(0, 0.5, len(current))
    output_path = tmp_path / "drift_report.html"

    dataset_drift_share, cate_drift_detected = build_drift_report(
        reference, current, reference_cate, current_cate, str(output_path)
    )

    assert dataset_drift_share == 0
    assert not cate_drift_detected
