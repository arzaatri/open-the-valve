import pandas as pd

from open_the_valve.causal.cate_estimators import CateRun, slice_cate_by_subgroup
from open_the_valve.causal.its import slice_its_by_subgroup

_CAVEATS = {
    "s_learner": "single shared model across arms; can under-fit heterogeneity when the "
    "treatment effect is small relative to outcome variance",
    "t_learner": "separate per-arm models; can overfit when the treated arm is small",
    "x_learner": "corrects T-learner's small-arm variance via propensity-weighted "
    "cross-imputation",
    "dr_learner": "doubly robust: consistent if either the propensity or outcome model "
    "is correctly specified",
    "linear_dml": "R-learner family; assumes a linear treatment effect in the covariates. "
    "Unlike CausalForestDML, its final stage has no regularization knob, so at this "
    "dataset's treated-arm size (~2% of rows) it is the least stable of the 6 -- a large "
    "swing here is itself informative: weight it below the regularized estimators",
    "causal_forest_dml": "R-learner family; nonparametric effect, most flexible of the 6",
    "its": "each game is its own control via its pre-treatment trend; immune to "
    "cross-game confounding but sensitive to the pre-window's own trend assumptions",
}


def build_comparison_table(cate_run: CateRun, its_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, result in cate_run.results.items():
        row = {"method": name, "ate": result.ate, "caveat": _CAVEATS[name]}
        for refuter_label, (estimated, new) in result.refutations.items():
            row[f"refute_{refuter_label}_estimated"] = estimated
            row[f"refute_{refuter_label}_new"] = new
        rows.append(row)

    rows.append(
        {
            "method": "its",
            "ate": (
                its_results["mean_post_effect"].mean() if not its_results.empty else float("nan")
            ),
            "caveat": _CAVEATS["its"],
        }
    )
    return pd.DataFrame(rows)


def build_cate_slice_table(
    cate_run: CateRun,
    its_results: pd.DataFrame,
    panel: pd.DataFrame,
    slice_dims: list[str],
    exploratory_dims: list[str],
) -> pd.DataFrame:
    all_slices = []
    for slice_col in [*slice_dims, *exploratory_dims]:
        is_exploratory = slice_col in exploratory_dims
        for result in cate_run.results.values():
            sliced = slice_cate_by_subgroup(result, cate_run, slice_col)
            sliced["is_exploratory"] = is_exploratory
            all_slices.append(sliced)

        its_sliced = slice_its_by_subgroup(its_results, panel, slice_col).rename(
            columns={"n_events": "n"}
        )
        its_sliced["estimator"] = "its"
        its_sliced["is_exploratory"] = is_exploratory
        all_slices.append(
            its_sliced[
                ["slice_value", "mean_effect", "n", "slice_dim", "estimator", "is_exploratory"]
            ]
        )

    return pd.concat(all_slices, ignore_index=True)


def write_findings_markdown(
    comparison_table: pd.DataFrame,
    slice_table: pd.DataFrame,
    causal_graph_dot: str,
    output_path: str,
) -> None:
    lines = ["# Open The Valve — Phase 2 Findings\n"]

    lines.append("## Method comparison\n")
    lines.append(comparison_table.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Causal graph (flat: every covariate is a confounder)\n")
    lines.append("```dot")
    lines.append(causal_graph_dot)
    lines.append("```\n")

    lines.append("## CATE by subgroup\n")
    lines.append(
        "Primary slices (genre, price_tier, depth_bucket) are point estimates for interpretation, "
        "not independently hypothesis-tested; exploratory slices (season, game_age_bucket) are "
        "descriptive only. See `is_exploratory` column.\n"
    )
    for slice_dim, group in slice_table.groupby("slice_dim"):
        lines.append(f"### {slice_dim}\n")
        n_by_slice = group.loc[group["estimator"] == "its", ["slice_value", "n"]]
        if not n_by_slice.empty:
            min_n, max_n = n_by_slice["n"].min(), n_by_slice["n"].max()
            lines.append(
                f"(events per slice value range {min_n}-{max_n}; treat any single-digit-N "
                "row as illustrative, not a reliable estimate)\n"
            )
        pivot = group.pivot_table(index="slice_value", columns="estimator", values="mean_effect")
        lines.append(pivot.to_markdown())
        lines.append("\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
