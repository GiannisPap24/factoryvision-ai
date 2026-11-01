"""Historical evidence presentation stays exact, scoped and inference-free."""

import json
from importlib.resources import files

from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import source_app as source_app


def test_cashew_metrics_match_published_card_and_thresholds(source_app):
    at, _ = source_app
    at.run().sidebar.radio[0].set_value("Research results").run()
    assert not at.exception
    card = files("factoryvision.app").joinpath("model_cards/model-card.md").read_text("utf-8")
    published = {
        cells[0]: cells[1:]
        for line in card.splitlines()
        if line.startswith("| ")
        for cells in [[cell.strip() for cell in line.split("|")[1:-1]]]
    }
    for table in at.table:
        for row in table.value.to_dict("records"):
            metric = row["Metric"].replace(" AP", " Average Precision")
            assert [row["Autoencoder"], row["Spatial mean"]] == published[metric]
    text = "\n".join(item.value for item in at.markdown)
    for threshold in ("0.0031531278509646654", "0.017185416119173163"):
        assert threshold in card and threshold in text
    assert any("now observed" in item.value for item in at.markdown)
    assert any("zero risk" in item.value for item in at.warning)
    assert any(
        "PCB1: BLOCKED" in item.value and "PCB2: INCONCLUSIVE" in item.value for item in at.warning
    )
    assert any("DEVELOPMENT ONLY" in item.value for item in at.info)
    assert any("Python 3.12.15" in item.value for item in at.caption)
    assert "results" not in at.session_state and not at.image


def test_metalparts_full_precision_matches_packaged_records(source_app):
    at, _ = source_app
    at.run().sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    at.sidebar.radio[0].set_value("Research results").run()
    assert not at.exception
    package = files("factoryvision.inference")
    comparison = json.loads(package.joinpath("metalparts_comparison.json").read_text())
    history = json.loads(package.joinpath("metalparts_research.json").read_text())
    ranking, timing, training, diagnostic = [table.value for table in at.table]
    for method, label in (("autoencoder", "Autoencoder"), ("feature_memory", "Feature memory")):
        assert ranking[label].tolist() == [
            str(comparison[method][level][metric])
            for level in ("image", "pixel")
            for metric in ("auroc", "average_precision")
        ]
        row = timing[timing.Method == label].iloc[0]
        assert row["Median CPU seconds"] == str(comparison[method]["latency_seconds"]["median"])
        assert row["p95 CPU seconds"] == str(comparison[method]["latency_seconds"]["p95"])
    for row, record in zip(training.to_dict("records"), history["training"]["epochs"], strict=True):
        assert row["Fitting MSE"] == str(record["train_loss"])
        assert row["Validation MSE"] == str(record["validation_loss"])
    for row in diagnostic.to_dict("records"):
        record = history["metrics"][row["Level"]]
        assert row["AUROC"] == str(record["auroc"])
        assert row["Average precision"] == str(record["average_precision"])
        assert row["Prevalence"] == str(record["prevalence"])
        assert row["Positive"] == record["positive"]
        assert row["Negative"] == record["negative"]
    assert any("NOT CLEARED" in item.value for item in at.warning)
    assert any("No binary threshold exists" in item.value for item in at.markdown)
    assert any("not a controlled speed comparison" in item.value for item in at.warning)
    assert any("approximate for unquantized" in item.value for item in at.caption)
    assert any("unavailable or failed verification" in item.value for item in at.info)
    assert not at.image and "metal_results" not in at.session_state
