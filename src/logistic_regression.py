import numpy
import math


class LogisticRegression:




    @staticmethod
    def log_odds(features, weights):
        """Estimates the log-odd ratio using linear regression (w0 +w1x1 +...+wmxm)
        
        Arguments:
            features {numpy.array dim=1} -- Data
            weights {numpy.array dim=1} -- Weights used in the linear regression model
        
        Raises:
            ValueError: If the dimension of the inputs is not 1 (row vector)
        
        Returns:
            float -- result of the log-odd ratio
        """
        # Check that features and weights are row vectors (i.e [[x, x, x, x, x ,x]])
        if (weights.ndim != 1 or features.ndim != 1):
            raise ValueError()
        # Calculate the log-odd
        w0 = weights[0]
        return w0 + numpy.dot(weights[1:], features)

    @staticmethod
    def logistic_func(features, weights):
        """Estimates the probability of the output being true using the logistic function

        logistic function = 1/(1 + exp(log-odd))
        
        Arguments:
            features {numpy.array dim=1} -- Data
            weights {numpy.array dim=1} -- Weights used in the linear regression model
        
        Returns:
            float -- probability of the output being true (from 0 to 1)
        """
        return 1/(1 + math.exp(LogisticRegression.log_odds(features, weights)))
