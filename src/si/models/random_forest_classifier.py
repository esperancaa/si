import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from si.data.dataset import Dataset

class RandomForestClassifier:
    """
    Random Forest Classifier: Combines multiple decision trees for classification
    """

    def __init__(self, n_estimators=1000, max_features=None, min_samples_split=2,
                 max_depth=10, mode='gini', seed=42):
        """
        Parameters:
        ----------
        n_estimators: int - Number of decision trees.
        max_features: int - Maximum features per tree.
        min_samples_split: int - Minimum samples to allow a split.
        max_depth: int - Maximum depth of the trees.
        mode: str - Criteria for splits ('gini' or 'entropy').
        seed: int - Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []  # Stores tuples (features, trained_tree)

    def fit(self, dataset):
        """
        Train the random forest classifier.

        Parameters:
        ----------
        dataset: Dataset - Input data to fit the model.

        Returns:
        -------
        self: RandomForestClassifier - Trained model.
        """
        np.random.seed(self.seed)  # Ensures reproducibility
        n_samples, n_features = dataset.shape()
        
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

            sampled_data = Dataset(dataset.X[sample_indices][:, feature_indices], dataset.y[sample_indices])

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_samples_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(sampled_data)
            self.trees.append((feature_indices, tree))

        return self

    def predict(self, dataset):
        """
        Predict class labels for input dataset.

        Parameters:
        ----------
        dataset: Dataset - Input data for prediction.

        Returns:
        -------
        np.ndarray - Predicted class labels.
        """
        n_samples = dataset.shape()[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object)

        for i, (feature_indices, tree) in enumerate(self.trees):
            sampled_data = Dataset(dataset.X[:, feature_indices], dataset.y)
            predictions[i, :] = tree.predict(sampled_data)

        # Voting: Most frequent label for each sample
        majority_vote = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)
        return majority_vote

    def score(self, dataset):
        """
        Calculate model accuracy on dataset.

        Parameters:
        ----------
        dataset: Dataset - Input data for evaluation.

        Returns:
        -------
        float - Accuracy score.
        """
        predictions = self.predict(dataset)
        return np.mean(predictions == dataset.y)