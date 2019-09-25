import numpy
from logistic_regression import LogisticRegression

FEATURES = numpy.ones(5, dtype=float)

WEIGHTS = numpy.arange(6, dtype=float)



def test_log_odds_valid_input():
    assert LogisticRegression.log_odds(FEATURES, WEIGHTS) == 15.0


def test_logistic_function_valid_input():
    features = numpy.zeros(5)
    weights = numpy.zeros(6)
    assert LogisticRegression.logistic_func(features, weights) == 0.5