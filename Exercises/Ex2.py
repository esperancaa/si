from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None, strategy: str= None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label
        self.strategy = strategy

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)
    
    def dropna(self) -> 'Dataset':
        """
        Removes samples with missing values

        Returns
        -------
        Dataset
        """
    
        mask = ~np.isnan(self.X).any(axis=1) # ~ inverte os valores do array booleano. Isso é feito porque queremos manter as linhas que não têm NaN
        
        if mask is not None:
            X = self.X[mask]
            y = self.y[mask] if self.has_label() else None
            return Dataset(X, y, features=self.features, label=self.label)
        
        
        else:
            return self
    
    def fillna(self) -> 'Dataset':
        """
        Fills missing values with a constant

        Parameters
        ----------
        value: int or float
            The value to fill missing values with

        Returns
        -------
        Dataset
        """
    
        mean_value = np.nanmean(self.X, axis=0)
        median_value= np.nanmedian(self.X, axis=0)
        value= 1
    
    
        if self.strategy == 'value':
            X = np.where(np.isnan(self.X), value, self.X)
            return Dataset(X, features=self.features, label=self.label)
        
        elif self.strategy == 'mean':
            X = np.where(np.isnan(self.X), mean_value, self.X)
            return Dataset(X, features=self.features, label=self.label)
    
        elif self.strategy == 'median':
            X = np.where(np.isnan(self.X), median_value, self.X)
            return Dataset(X, features=self.features, label=self.label)
    
        else:
            raise ValueError("Invalid strategy")
    
    def remove_by_index(self, index: int) -> 'Dataset':
        """
        Removes samples by index

        Parameters
        ----------
        index: int or list of int
            The index or indices to remove

        Returns
        -------
        Dataset
        """
        if not isinstance(index, int):
                raise ValueError("Please provide a valid integer index.")
        
        if index <0 and index> self.X.shape[0]:
                raise ValueError("Put a valid index")
            
        self.X = np.delete(self.X, index, axis=0) #remover a posição index do eixo (linhas do dataset X) 
        if self.y is not None: #caso exista y vou quere apagar tb o seu index
            self.y = np.delete(self.y, index)
        return self
    
        

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)