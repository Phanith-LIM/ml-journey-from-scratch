import numpy as np

class RegressionMetrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(RegressionMetrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def r_squared(y_true, y_pred):
        mean_y = np.mean(y_true)
        ss_total = np.sum((y_true - mean_y) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
