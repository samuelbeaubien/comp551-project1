import numpy
from logistic_regression import LogisticRegression
from unittest import mock


def test_log_odds_valid_input():
    features = numpy.ones(5, dtype=float)
    weights = numpy.arange(5, dtype=float)
    assert LogisticRegression.log_odds(features, weights) == 10.0


def test_logistic_function_valid_input():
    """Tests that when the log-odd function returns 0, the logistic function returns 0.5
    """
    features = numpy.zeros(5)
    weights = numpy.zeros(5)

    with mock.patch('logistic_regression.LogisticRegression.log_odds') as mock_log_odds:
        mock_log_odds.return_value = 0
        assert LogisticRegression.logistic_func(features, weights) == 0.5


def test_gradient_descent():
    #Get inputs
    inputs = numpy.array([
                            [1, 0],
                            [2, 0], 
                            [3, 0], 
                            [4, 0], 
                            [5, 0],
                            [6, 1], 
                            [7, 1],
                            [8, 1],
                            [9, 1],
                            [10, 1],
                         ])
    
    weights = numpy.array([1.0, 1.0])
    step_size = 0.001
    max_iterations = 1000000
    max_w_diff = 0.01

    weights = LogisticRegression.gradient_descent(inputs, weights, step_size, max_iterations, max_w_diff)

    assert weights == None