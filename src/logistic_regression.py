import numpy
import math


class LogisticRegression:




    @staticmethod
    def log_odds(features, weights):
        """Estimates the log-odd ratio using linear regression (w0 +w1x1 +...+wmxm)
        
        Arguments:
            features {numpy.ndarray dim=1} -- Data
            weights {numpy.ndarray dim=1} -- Weights used in the linear regression model
        
        Raises:
            ValueError: If the dimension of the inputs is not 1 (row vector)
        
        Returns:
            float -- result of the log-odd ratio
        """
        # Check that features and weights are row vectors (i.e [[x, x, x, x, x ,x]])
        if (weights.ndim != 1 or features.ndim != 1):
            raise ValueError()
        # Calculate the log-odd
        return numpy.dot(weights, features)

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
        return 1/(1 + math.exp(LogisticRegression.log_odds(features, weights)))

    @staticmethod
    def fit(data_set, weights, step_size, max_iterations, max_w_diff):
        """ wk+1 = wk + αk ∑i=1:n xi (yi – σ(wkTxi))
            
        
        Arguments:      
            data_set {numpy.ndarray dim = 2} -- Set of different training examples.
            weights {numpy.ndarray dim = 1} -- Weights used in the linear regression.
            step_size {float} -- Step size of the gradient descent.
            max_iterations {int} -- Maximum number of iterations that the gradient descent algorithm
                will perform.
            max_w_diff {float} -- Max weight difference allowed in order to terminate the algorithm.

        """
        # Make sure that all arguments are floats
        data_set = data_set.astype(float)
        weights = weights.astype(float)
        
        # Prepare initial conditions to ensure at least one iteration
        current_weights = weights
        old_weights = None
        num_iterations = 0
        w_diff = max_w_diff*2
        
        while (num_iterations < max_iterations) and (w_diff > max_w_diff):
            old_weights = current_weights
            update_sum = numpy.zeros_like(old_weights)

            # ∑i=1:n xi (yi – σ(wkTxi))
            for row in data_set:
                # Get outcome (y)
                y = row[-1] 
                # Get the features
                features = row[:-1]
                # Add intercept term of value "1" so that features and weights vectors have the same 
                # length when adding them together
                features = numpy.concatenate(([1.0], features))
                # Compute gradient with features
                update_sum += features * (y - LogisticRegression.logistic_func(features, weights))
            
            # Add the step_size
            update_sum =  update_sum * step_size
            current_weights = old_weights + update_sum
            num_iterations += 1
            w_diff = abs(numpy.sum(old_weights - current_weights))
        return current_weights

            
