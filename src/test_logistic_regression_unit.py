import numpy as np
from logistic_regression import LogisticRegression
from unittest import mock


def test_log_odds_valid_input():
    features = np.ones(5, dtype=float)
    weights = np.arange(5, dtype=float)
    assert LogisticRegression.log_odds(features, weights) == 10.0


def test_logistic_function_valid_input():
    """Tests that when the log-odd function returns 0, the logistic function returns 0.5
    """
    features = np.zeros(5)
    weights = np.zeros(5)

    with mock.patch('logistic_regression.LogisticRegression.log_odds') as mock_log_odds:
        mock_log_odds.return_value = 0
        assert LogisticRegression.logistic_func(features, weights) == 0.5


# def test_gradient_descent():
#     #Get inputs
#     x = np.arange(10, dtype=float)
#     x = x.reshape(10, 1)

#     y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#     weights = np.array([1.0, 1.0])
#     step_size = 0.001
#     max_iterations = 1000
#     max_w_diff = 0.01

#     weights = LogisticRegression.fit(x, y, step_size, max_iterations, max_w_diff)

#     assert weights == None


def test_actual_data():

    #import data from csv (wine)/text (breast), and drop first column in breast
    data = np.genfromtxt("winequality-red.csv", delimiter=';', skip_header=1)
    data2 = np.genfromtxt("breast-cancer-wisconsin.data", delimiter = ',')
    data2 = data2[:, 1:]

    #print data and dimensions
    r = data.shape[0]
    c = data.shape[1]
    r2 = data2.shape[0]
    c2 = data2.shape[1]

    #turn labels into binary (>=6 --> 1)/(<6 --> 0)
    i = 0
    while i < r:
        if data[i,c-1] >= 6:
            
            data[i,c-1] = 1
        else:
            data[i,c-1] = 0
        i = i+1
    #turn labels into binary (4 --> 1)/(2 --> 0)
    i = 0
    while i < r2:
        if data2[i,c2-1] == 2:
            data2[i,c2-1] = 0
        else:
            data2[i,c2-1] = 1
        i = i+1

    #clean data from incorrect entries
    data = data[~np.isnan(data).any(axis=1)]
    data2 = data2[~np.isnan(data2).any(axis=1)]

    r = data.shape[0]
    c = data.shape[1]
    r2 = data2.shape[0]
    c2 = data2.shape[1]

    for i in range (0,r):
        for j in range (0,c):
            if type(data[i,j]) != np.float64:
                print("error")

    for i in range (0,r2):
        for j in range (0,c2):
            if type(data2[i,j]) != np.float64:
                print("error")

    #Seperate labels from features and print dimensions
    X = data[:, 0:c-1]
    Y = data[:, c-1:c]
    Y = Y.ravel()

    X2 = data2[:, 0:c2-1]
    Y2 = data2[:, c2-1:c]

    # Code

    model = LogisticRegression()

    out = model.fit(X,Y,0.0001, 100, 0.001)

    assert 1 == None

    







