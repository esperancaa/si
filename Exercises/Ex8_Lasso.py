import numpy as np
from si.data.dataset import Dataset
class LassoRegression:
    def __init__(self, l1_penalty=1.0, max_iter=1000, tolerance=1e-4):
        """
        Lasso Regression implementation using coordinate descent.

        Parameters
        ----------
        l1_penalty : float
            The regularization parameter (lambda).
        max_iter : int
            Maximum number of iterations for optimization.
        tolerance : float
            Convergence tolerance for stopping criteria.
        """
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, rho, lam):
        """
        Apply the soft-thresholding operator for Lasso.

        Parameters
        ----------
        rho : float
            The correlation between feature and residual.
        lam : float
            The regularization parameter (lambda).

        Returns
        -------
        float
            The updated coefficient value after soft-thresholding.
        """
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0

    def fit(self, dataset):
        """
        Fit the Lasso Regression model.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object containing features (X) and labels (y).
        """
        X, y = dataset.X, dataset.y
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        for iteration in range(self.max_iter):
            coef_prev = self.coef_.copy()

            # Update intercept
            self.intercept_ = np.mean(y - np.dot(X, self.coef_))

            # Update each coefficient
            for j in range(n_features):
                residual = y - (np.dot(X, self.coef_) + self.intercept_)
                rho = np.dot(X[:, j], residual + self.coef_[j] * X[:, j])
                self.coef_[j] = self._soft_threshold(rho, self.l1_penalty) / np.sum(X[:, j] ** 2)

            # Check for convergence
            if np.sum(np.abs(self.coef_ - coef_prev)) < self.tolerance:
                break

    def predict(self, dataset):
        """
        Predict target values using the fitted Lasso model.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object containing features (X).

        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted target values.
        """
        X = dataset.X
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, dataset):
        """
        Calculate the Mean Squared Error (MSE) of the predictions.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object containing features (X) and labels (y).

        Returns
        -------
        float
            The Mean Squared Error of the predictions.
        """
        X, y = dataset.X, dataset.y
        y_pred = self.predict(dataset)
        return np.mean((y - y_pred) ** 2)

