from typing import Tuple

import numpy as np

from si.data.dataset import Dataset

from sklearn.utils import shuffle


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test



import numpy as np
from sklearn.utils import shuffle
from typing import Tuple
import numpy as np
from si.data.dataset import Dataset

def stratified_train_test_split(dataset:Dataset, test_size=0.2, random_state:int =None) ->Tuple[Dataset, Dataset]:
    """
    split the dataset into training and testing sets while maintaining the class distribution
    
    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator
        
    Returns
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    
    """
    
    X= dataset.X
    y= dataset.y
    
    labels = y
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    train= []
    test=[]
    if random_state is not None:
        np.random.seed(random_state)
        
    for label, count in zip(unique_classes, class_counts):
        
        idxs = np.where(labels == label)[0]
        
        num_test= int(np.floor(test_size * count))
        
        idxs= shuffle(idxs, random_state= random_state)
        
        lables_test_idxs= idxs[:num_test]
        test.extend(lables_test_idxs) #use the extendo because we add multiple elements
        
        lables_train_idxs= idxs[num_test:]
        train.extend(lables_train_idxs)
    
    train= np.array(train,dtype=int)
    test= np.array(test,dtype=int)
    
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    
    train_dataset = {'data': X_train, 'target': y_train}
    test_dataset = {'data': X_test, 'target': y_test}
    
    train_dataset = Dataset(X_train, y_train, features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X_test, y_test, features=dataset.features, label=dataset.label)
    

    return train_dataset, test_dataset