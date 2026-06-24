from app.agent.executor import _safe_eval


def test_safe_calculator():
    """
    Test that the safe mathematical evaluator performs valid math and returns float/int results.
    """
    assert _safe_eval("1500 * (1.1 + 0.05)") == 1725.0
    assert _safe_eval("10 / 2 - 1") == 4.0
