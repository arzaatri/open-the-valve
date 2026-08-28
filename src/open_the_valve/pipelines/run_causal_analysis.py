import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import select

from open_the_valve.causal.cate_estimators import build_causal_graph, fit_all_estimators
from open_the_valve.causal.its import run_its_all_events
from open_the_valve.config_models import AppConfig
from open_the_valve.db.models import DiscountEvent
from open_the_valve.db.session import make_engine
from open_the_valve.io_utils.hydra_entrypoint import hydra_entrypoint
from open_the_valve.reports.findings import (
    build_cate_slice_table,
    build_comparison_table,
    write_findings_markdown,
)

load_dotenv()
logger = logging.getLogger(__name__)

_OUTCOME_COL = "log_player_count"
_TREATMENT_COL = "is_discounted"
_CONFOUNDERS = ["genre", "price_tier", "depth_pct", "day_of_week", "is_platform_sale_window"]
# Narrower than _CONFOUNDERS on purpose: with only ~2% of rows treated, fitting
# CATE heterogeneity (X) over the full one-hot confounder set is unstable (see
# cate_estimators.build_causal_model's docstring) -- X is restricted to the
# dimensions actually reported on in the CATE slice table.
_HETEROGENEITY_COVARIATES = ["genre", "price_tier", "depth_pct"]
_TREATMENT_STORE = "Steam"


def run(config: AppConfig) -> None:
    panel = pd.read_parquet(config.causal.panel.output_path)
    panel[_TREATMENT_COL] = panel[_TREATMENT_COL].astype(int)

    engine = make_engine(config.db)
    discount_events = pd.read_sql(
        select(DiscountEvent.game_id, DiscountEvent.store, DiscountEvent.start_at).where(
            DiscountEvent.store == _TREATMENT_STORE
        ),
        engine,
    )

    logger.info("running ITS across %d Steam discount events", len(discount_events))
    its_results = run_its_all_events(
        panel, discount_events, config.causal.its, outcome_col=_OUTCOME_COL
    )

    logger.info("fitting 6 EconML estimators via DoWhy")
    cate_run = fit_all_estimators(
        panel,
        _OUTCOME_COL,
        _TREATMENT_COL,
        _CONFOUNDERS,
        _HETEROGENEITY_COVARIATES,
        config.causal.cate,
    )

    comparison_table = build_comparison_table(cate_run, its_results)
    slice_table = build_cate_slice_table(
        cate_run,
        its_results,
        panel,
        config.causal.cate.slice_dims,
        config.causal.cate.exploratory_dims,
    )
    causal_graph = build_causal_graph(_CONFOUNDERS, _TREATMENT_COL, _OUTCOME_COL)

    output_dir = os.path.dirname(config.causal.findings.output_path)
    os.makedirs(output_dir, exist_ok=True)
    comparison_table.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)
    slice_table.to_csv(os.path.join(output_dir, "cate_slices.csv"), index=False)
    write_findings_markdown(
        comparison_table, slice_table, causal_graph, config.causal.findings.output_path
    )

    logger.info("wrote findings to %s", config.causal.findings.output_path)


main = hydra_entrypoint(run)

if __name__ == "__main__":
    main()
