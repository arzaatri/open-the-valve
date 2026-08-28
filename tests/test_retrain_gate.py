from open_the_valve.pipelines.retrain_if_needed import should_retrain


def test_should_retrain_when_no_prior_run():
    assert should_retrain(None, current_row_count=100, threshold=50)


def test_should_retrain_at_or_above_threshold():
    assert should_retrain(last_panel_row_count=1000, current_row_count=1050, threshold=50)
    assert should_retrain(last_panel_row_count=1000, current_row_count=1100, threshold=50)


def test_should_not_retrain_below_threshold():
    assert not should_retrain(last_panel_row_count=1000, current_row_count=1049, threshold=50)
    assert not should_retrain(last_panel_row_count=1000, current_row_count=1000, threshold=50)
