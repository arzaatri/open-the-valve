import numpy as np
import pandas as pd
import pytest

from open_the_valve.causal.cate_estimators import fit_all_estimators, slice_cate_by_subgroup
from open_the_valve.config_models import CateConfig


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    genre = rng.choice(["Shooter", "Strategy"], size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    outcome = x1 + treatment * (2 + x1) + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "genre": genre, "is_discounted": treatment, "outcome": outcome})


@pytest.fixture(scope="module")
def config() -> CateConfig:
    return CateConfig(
        xgb_n_estimators=20,
        xgb_max_depth=2,
        xgb_learning_rate=0.3,
        cf_n_estimators=40,
        cf_min_samples_leaf=5,
        estimator_names=[
            "s_learner",
            "t_learner",
            "x_learner",
            "dr_learner",
            "linear_dml",
            "causal_forest_dml",
        ],
        slice_dims=["genre"],
        exploratory_dims=[],
        refutation_num_simulations=3,
        refutation_seed=0,
    )


@pytest.fixture(scope="module")
def cate_run(synthetic_panel, config):
    return fit_all_estimators(
        synthetic_panel, "outcome", "is_discounted", ["x1", "genre"], ["genre"], config
    )


def test_all_six_estimators_fit_with_finite_ate(cate_run, config):
    assert set(cate_run.results.keys()) == set(config.estimator_names)
    for result in cate_run.results.values():
        assert np.isfinite(result.ate)


def test_all_six_estimators_slice_without_error(cate_run):
    for result in cate_run.results.values():
        sliced = slice_cate_by_subgroup(result, cate_run, "genre")
        assert set(sliced["slice_value"]) == {"Shooter", "Strategy"}
        assert sliced["mean_effect"].notna().all()


def test_causal_forest_dml_refutation_pipeline_runs_end_to_end(cate_run):
    result = cate_run.results["causal_forest_dml"]
    assert set(result.refutations.keys()) == {
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
    }
    for estimated_effect, new_effect in result.refutations.values():
        assert np.isfinite(estimated_effect)
        assert np.isfinite(new_effect)
