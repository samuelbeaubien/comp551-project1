import numpy as np
import math

class LogisticRegression:

    @staticmethod
    def log_odds(features, weights):
        """Estimates the log-odd ratio using linear regression (w0 +w1x1 +...+wmxm)
        
        Arguments:
            features {numpy.ndarray dim=1} -- Data
            weights {numpy.ndarray dim=1} -- Weights used in the linear regression model
        
        Raises:
            ValueError: If the dimension of the inputs is not 1 (cur_row vector)
        
        Returns:
            float -- result of the log-odd ratio
        """
        # Check that features and weights are cur_row vectors (i.e [[x, x, x, x, x ,x]])
        if (weights.ndim != 1 or features.ndim != 1):
            raise ValueError()
        # Calculate the log-odd
        return np.dot(weights, features)

    @staticmethod
    def logistic_func(features, weights):
        """Estimates the probability of the output being true using the logistic function

        logistic function = 1/(1 + exp(log-odd))
        
        Arguments:
            features {numpy.ndarray dim=1} -- Data
            weights {numpy.ndarray dim=1} -- Weights used in the linear regression model
        
        Returns:
            float -- probability of the output being true (from 0 to 1)
        """
        try:
            return 1/(1 + math.exp(LogisticRegression.log_odds(features, weights)))
        except:
            # In case of overflow, the log-odds is basically 0
            return 1/(1 + math.exp(0))

    @staticmethod
    def fit(x_features, y_outcomes, learning_rate, max_iterations, max_iter_diff):
        """ Fit using logistic regression
        
        wk+1 = wk + αk ∑i=1:n xi (yi – σ(wkTxi))
            
        Arguments:      
            x_features {numpy.ndarray dim = 2} -- Set of different training examples.
            y_outcomes {numpy.ndarray dim = 1} -- Outcomes
            weights {numpy.ndarray dim = 1} -- Weights used in the linear regression.
            learning_rate {float} -- Step size of the gradient descent.
            max_iterations {int} -- Maximum number of iterations that the gradient descent algorithm
                will perform.
            max_iter_diff {float} -- Max weight difference allowed in order to terminate the algorithm.

        """
        weights = np.ones(x_features.shape[1] + 1)
        # Make sure that all arguments are floats
        x_features = x_features.astype(float)
        y_outcomes = y_outcomes.astype(float)
        weights = weights.astype(float)
        # Prepare initial conditions to ensure at least one iteration
        cur_weights = weights
        old_weights = None
        num_iterations = 0
        iter_diff = max_iter_diff*2
        # Every iterations modifies the weights
        while (num_iterations < max_iterations) and (iter_diff > max_iter_diff):
            old_weights = cur_weights
            update_sum = np.zeros_like(old_weights)
            # ∑ xi(yi – σ(wkTxi))
            for cur_row,cur_y in zip(x_features, y_outcomes):
                # Add intercept term of value "1" so that features and weights vectors have same length when adding them together
                cur_row = np.concatenate(([1.0], cur_row))
                update_sum += cur_row * (cur_y - LogisticRegression.logistic_func(cur_row, old_weights))
            # Add the learning_rate
            cur_weights = old_weights - (update_sum * learning_rate)
            num_iterations += 1
            iter_diff = abs(np.sum(old_weights - cur_weights))
        return cur_weights

            
