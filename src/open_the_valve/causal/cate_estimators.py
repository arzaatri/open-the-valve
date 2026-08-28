import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dowhy import CausalModel
from xgboost import XGBClassifier, XGBRegressor

from open_the_valve.config_models import CateConfig

logger = logging.getLogger(__name__)

_ESTIMATOR_METHOD_NAMES = {
    "s_learner": "backdoor.econml.metalearners.SLearner",
    "t_learner": "backdoor.econml.metalearners.TLearner",
    "x_learner": "backdoor.econml.metalearners.XLearner",
    "dr_learner": "backdoor.econml.dr.DRLearner",
    "linear_dml": "backdoor.econml.dml.LinearDML",
    "causal_forest_dml": "backdoor.econml.dml.CausalForestDML",
}
_REFUTER_METHOD_NAMES = {
    "placebo_treatment": "placebo_treatment_refuter",
    "random_common_cause": "random_common_cause",
    "data_subset": "data_subset_refuter",
}


@dataclass
class EstimatorResult:
    name: str
    ate: float
    refutations: dict[str, tuple[float, float]]  # refuter label -> (estimated_effect, new_effect)
    fitted_estimator: Any  # underlying EconML object, exposes .effect(X) for CATE


@dataclass
class CateRun:
    results: dict[str, EstimatorResult]
    data: (
        pd.DataFrame
    )  # rows actually used for fitting (post dropna), original covariate columns intact
    encoded_covariates: pd.DataFrame  # heterogeneity-only (X) encoding, same row order as `data`
    encoded_confounders: (
        pd.DataFrame
    )  # full confounder (W ∪ X) encoding -- see slice_cate_by_subgroup


def build_causal_graph(covariates: list[str], treatment_col: str, outcome_col: str) -> str:
    """DOT-format documentation of the causal structure -- every covariate is a
    confounder (edge into both treatment and outcome). This is for the findings
    doc / logging only; build_causal_model constructs the CausalModel via
    common_causes + effect_modifiers directly rather than parsing this graph
    (see build_causal_model's docstring for why).
    """
    lines = [f'"{treatment_col}" -> "{outcome_col}";']
    for cov in covariates:
        lines.append(f'"{cov}" -> "{treatment_col}"; "{cov}" -> "{outcome_col}";')
    return "digraph {\n  " + "\n  ".join(lines) + "\n}"


def build_causal_model(
    data: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    confounders: list[str],
    heterogeneity_covariates: list[str],
) -> tuple[CausalModel, object]:
    """Builds the dowhy.CausalModel and calls identify_effect() once; the
    returned estimand is reused across all 6 estimator fits.

    `confounders` (W) enters the backdoor adjustment set for every estimator.
    `heterogeneity_covariates` (X) is a narrower subset used for CATE
    heterogeneity -- kept small deliberately: fitting a high-dimensional
    heterogeneous-effect surface (one-hot day_of_week etc.) from only ~2% of
    rows being treated is unstable (verified empirically: LinearDML/
    CausalForestDML ATEs were off by orders of magnitude with the full
    confounder set as X; narrowing X to the actual slice dimensions fixed it).
    Passing a DOT graph= instead of common_causes/effect_modifiers causes
    DoWhy to re-derive both structurally from the graph, which collapses
    effect_modifiers to empty for a flat all-confounders graph and breaks
    CausalForestDML/LinearDML with "This estimator does not support X=None!".
    """
    model = CausalModel(
        data=data,
        treatment=treatment_col,
        outcome=outcome_col,
        common_causes=confounders,
        effect_modifiers=heterogeneity_covariates,
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model, identified_estimand


def _xgb_init_params(name: str, config: CateConfig) -> dict:
    kwargs = dict(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        random_state=config.refutation_seed,
    )
    if name == "s_learner":
        return {"overall_model": XGBRegressor(**kwargs)}
    if name in ("t_learner", "x_learner"):
        return {"models": XGBRegressor(**kwargs)}
    if name == "dr_learner":
        return {
            "model_propensity": XGBClassifier(**kwargs),
            "model_regression": XGBRegressor(**kwargs),
            "model_final": XGBRegressor(**kwargs),
        }
    if name == "linear_dml":
        return {
            "model_y": XGBRegressor(**kwargs),
            "model_t": XGBClassifier(**kwargs),
            "discrete_treatment": True,
            "random_state": config.refutation_seed,
        }
    if name == "causal_forest_dml":
        # Forest-level regularization (min_samples_leaf) is what actually
        # stabilizes this estimator at low treated-N -- EconML's defaults are
        # tuned for much larger samples than the ~84 treated rows here.
        return {
            "model_y": XGBRegressor(**kwargs),
            "model_t": XGBClassifier(**kwargs),
            "discrete_treatment": True,
            "random_state": config.refutation_seed,
            "n_estimators": config.cf_n_estimators,
            "min_samples_leaf": config.cf_min_samples_leaf,
        }
    raise ValueError(f"unknown estimator name: {name}")


def _run_refutations(
    model: CausalModel, identified_estimand: object, estimate: object, config: CateConfig
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    for label, refuter_name in _REFUTER_METHOD_NAMES.items():
        kwargs: dict = {
            "num_simulations": config.refutation_num_simulations,
            "random_state": config.refutation_seed,
        }
        if refuter_name == "placebo_treatment_refuter":
            kwargs["placebo_type"] = "permute"
        if refuter_name == "data_subset_refuter":
            kwargs["subset_fraction"] = 0.8
        try:
            refutation = model.refute_estimate(
                identified_estimand, estimate, method_name=refuter_name, **kwargs
            )
            results[label] = (float(refutation.estimated_effect), float(refutation.new_effect))
        except Exception:
            logger.exception("refuter %s failed", refuter_name)
            results[label] = (float("nan"), float("nan"))
    return results


def fit_estimator(
    model: CausalModel, identified_estimand: object, name: str, config: CateConfig
) -> EstimatorResult:
    estimate = model.estimate_effect(
        identified_estimand,
        method_name=_ESTIMATOR_METHOD_NAMES[name],
        control_value=0,
        treatment_value=1,
        target_units="ate",
        confidence_intervals=False,
        method_params={"init_params": _xgb_init_params(name, config), "fit_params": {}},
    )
    refutations = _run_refutations(model, identified_estimand, estimate, config)
    return EstimatorResult(
        name=name,
        ate=float(estimate.value),
        refutations=refutations,
        fitted_estimator=estimate.estimator.estimator,
    )


def fit_all_estimators(
    panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    confounders: list[str],
    heterogeneity_covariates: list[str],
    config: CateConfig,
) -> CateRun:
    """Encodes categorical covariates to numeric (all 6 estimators need a
    purely numeric X/W), builds one CausalModel/estimand shared by all 6, and
    fits every estimator in config.estimator_names against it.

    heterogeneity_covariates must be a subset of confounders -- their encoded
    (one-hot) columns are pulled out of the same encoded confounder frame so X
    and W are consistently derived from one encoding pass.
    """
    data = panel.dropna(subset=[outcome_col, treatment_col, *confounders]).reset_index(drop=True)
    for col in confounders:
        if data[col].dtype == bool:
            data[col] = data[col].astype(int)
    categorical_cols = [c for c in confounders if not pd.api.types.is_numeric_dtype(data[c])]
    encoded_confounders = pd.get_dummies(
        data[confounders], columns=categorical_cols, drop_first=True
    ).astype(float)

    hetero_prefixes = tuple(f"{h}_" for h in heterogeneity_covariates)
    hetero_encoded_cols = [
        c
        for c in encoded_confounders.columns
        if c in heterogeneity_covariates or c.startswith(hetero_prefixes)
    ]
    encoded_heterogeneity = encoded_confounders[hetero_encoded_cols]

    model_df = pd.concat([data[[outcome_col, treatment_col]], encoded_confounders], axis=1)
    model, identified_estimand = build_causal_model(
        model_df,
        outcome_col,
        treatment_col,
        encoded_confounders.columns.tolist(),
        hetero_encoded_cols,
    )

    results = {}
    for name in config.estimator_names:
        logger.info("fitting %s", name)
        results[name] = fit_estimator(model, identified_estimand, name, config)

    return CateRun(
        results=results,
        data=data,
        encoded_covariates=encoded_heterogeneity,
        encoded_confounders=encoded_confounders,
    )


def slice_cate_by_subgroup(
    result: EstimatorResult, cate_run: CateRun, slice_col: str
) -> pd.DataFrame:
    """Per-level mean CATE for one fitted estimator, sliced by a raw covariate
    column in cate_run.data (may or may not be one of the model's own encoded
    covariates -- exploratory dims like season/game_age_bucket don't need to
    have been in the model to be sliced on here).

    S/T/X-learner have no separate W input in EconML's API, so DoWhy fits them
    on the full confounder+heterogeneity matrix rather than the heterogeneity
    subset alone -- .effect() on those needs encoded_confounders, not
    encoded_covariates. DR-learner/LinearDML/CausalForestDML take the narrower
    heterogeneity-only X. Tried narrow-first since that's the common case, with
    a fallback rather than hardcoding per-estimator-name branching here.
    """
    try:
        X = cate_run.encoded_covariates.to_numpy()
        per_row_cate = result.fitted_estimator.effect(X)
    except ValueError:
        X = cate_run.encoded_confounders.to_numpy()
        per_row_cate = result.fitted_estimator.effect(X)
    sliced = pd.DataFrame(
        {"slice_value": cate_run.data[slice_col].to_numpy(), "cate": per_row_cate}
    )
    out = (
        sliced.groupby("slice_value", observed=True)["cate"]
        .agg(mean_effect="mean", n="count")
        .reset_index()
    )
    out["slice_dim"] = slice_col
    out["estimator"] = result.name
    return out
