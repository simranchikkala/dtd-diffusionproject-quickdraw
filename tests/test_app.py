"""
End-to-end tests for the Streamlit app, using Streamlit's built-in AppTest
(runs app.py without a browser). Skipped if model.pt hasn't been trained yet.
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
needs_model = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "model.pt")),
                                 reason="model.pt not trained yet")


def run_app(**changes):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if "source" in changes:
        at.radio[0].set_value(changes["source"]).run()
    if "example" in changes:
        at.selectbox[0].set_value(changes["example"]).run()
    return at


@needs_model
def test_app_starts_with_no_drawing():
    at = run_app()
    assert not at.exception
    assert any("Draw a star or a moon" in i.value for i in at.info)
    # The About tab is still rendered even though there is no drawing yet.
    assert any("How it works" in h.value for h in at.header)


@needs_model
@pytest.mark.parametrize("example", ["Star (outline)", "Star (one stroke)", "Moon"])
def test_app_runs_on_examples(example):
    at = run_app(source="Example", example=example)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Compare with the baselines" in s for s in subheaders)
    assert any("MSE" in m.value and "SSIM" in m.value for m in at.markdown)


@needs_model
def test_app_settings_change_without_errors():
    at = run_app(source="Example", example="Moon")
    at.sidebar.slider[0].set_value(0.3)   # noise
    at.sidebar.slider[2].set_value(2.0)   # guidance
    at.run()
    assert not at.exception
