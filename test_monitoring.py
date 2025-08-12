import os
import sys
from unittest.mock import MagicMock, patch
from prometheus_client import CollectorRegistry, Gauge

os.environ["METRICS_NAMESPACE"] = "testns"
os.environ["METRICS_SUBSYSTEM"] = "testsub"

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot/routes/utilities"))
import monitoring

original_blue_score = monitoring.blue_score

def print_and_call_real_blue_score(*args, **kwargs):
    print(">>> blue_score() called in test")
    
    kwargs['metric_namespace'] = "testns"
    kwargs['metric_subsystem'] = "testsub"

    registry = CollectorRegistry()


    with patch('monitoring.Gauge', side_effect=lambda *a, **kw: Gauge(*a, registry=registry, **kw)):
        func = original_blue_score(*args, **kwargs)
    return func


def test_blue_score_with_print_wrap():
    with patch("monitoring.blue_score", side_effect=print_and_call_real_blue_score):
        instrumentation_fn = monitoring.blue_score(
            metric_namespace="testns",
            metric_subsystem="testsub"
        )

        mock_info = MagicMock()
        mock_info.modified_handler = "/api/v1/feedback"
        mock_info.response.headers = {"BLEU": "0.75"}

        print("\n>>> Calling instrumentation function ")
        instrumentation_fn(mock_info)
        print("\n>>> instrumentation function completed")


def test_blue_score_sets_metric():
    registry = CollectorRegistry()

    def gauge_side_effect(*args, **kwargs):
        return Gauge(*args, registry=registry, **kwargs)

    with patch("monitoring.Gauge", side_effect=gauge_side_effect) as MockGauge:
        instrumentation_fn = monitoring.blue_score(
            metric_namespace="testns",
            metric_subsystem="testsub"
        )

        mock_info = MagicMock()
        mock_info.modified_handler = "/api/v1/feedback"
        mock_info.response.headers = {"BLEU": "0.75"}

        instrumentation_fn(mock_info)

        gauge_instance = MockGauge.return_value

        print("\n>>> test_blue_score_sets_metric: Gauge called with args:", MockGauge.call_args)
        MockGauge.assert_called_once_with(
            "BLEU_score",
            "bleu score metric",
            namespace="testns",
            subsystem="testsub"
        )

def test_blue_score_no_bleu_header_does_not_set():
    registry = CollectorRegistry()
    with patch("monitoring.Gauge", side_effect=lambda *a, **kw: Gauge(*a, registry=registry, **kw)) as MockGauge:
        mock_gauge_instance = MagicMock()
        MockGauge.return_value = mock_gauge_instance

        instrumentation_fn = monitoring.blue_score()

        mock_info = MagicMock()
        mock_info.modified_handler = "/api/v1/feedback"
        mock_info.response.headers = {}

        instrumentation_fn(mock_info)

        print("\n>>> test_blue_score_no_bleu_header_does_not_set: set called?", mock_gauge_instance.set.called)

        mock_gauge_instance.set.assert_not_called()


def test_instrumentator_has_metrics_added():
    print("\n>>> instrumentator has add method:", hasattr(monitoring.instrumentator, "add"))
    assert hasattr(monitoring.instrumentator, "add")
    assert monitoring.instrumentator is not None
