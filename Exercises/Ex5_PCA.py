from base import transformer
from numpy import np
from data import dataset
from matplotlib import plt

class PCA (transformer):
    """
    Principal Component Analysis (PCA) is a linear algebra technique used to reduce the dimensions
    of the dataset. The PCA to be implemented must use eigenvalue
    decomposition of the covariance matrix of the data.
    
    Parameters:
    -----------
    
    n_components: int, default=2
    
    Returns:
    -----------
    
    PCA object
    
    """
    
    def __init__(self, n_components=2):
        
        self.n_components = n_components
        self.components: np.ndarray = None
        self.mean: np.ndarray = None
        self.explained_variance: np.ndarray = None
        self.eigen_values= None
        self.eigen_vectors= None
    
    def _fit(self, dataset: dataset)-> tuple:
        """
        Fit the PCA model to the data
        
        Parameters:
        -----------
        
        X: numpy array
            The data to fit the PCA model to
            
        Returns:
        -----------
        
        self: PCA
            The fitted PCA model
            
        """
        
        if self.n_components >= self.dataset.X.shape[1]:
            raise ValueError("The number of components should be less than the number of features in the dataset.")

        
        self.mean= np.mean(dataset.X, axis=0)
        X_centered= np.subtract(dataset.X, self.mean) #centre the points
        
        covariance= np.cov(X_centered, rowvar=False)# covariance matrix
        
        self.eigen_values, self.eigen_vectors= np.linalg.eig(covariance)# eigenvalue decomposition
        
        sorted_indices = np.argsort(self.eigen_values)[::-1]
        eigen_values_sorted = self.eigen_values[sorted_indices]
        eigen_vectors_sorted = self.eigen_vectors[:, sorted_indices]
        
        self.components= eigen_vectors_sorted[:, :self.n_components]
        
        self.explained_variance = eigen_values_sorted[:self.n_components] / np.sum(self.eigen_values)

        return self
        
    def _transform (self, dataset:dataset):
        
        """
        
        ----------
        dataset (Dataset): Dataset object
        """
        X_centered= np.substract(dataset.X, self.mean)
        
        X_reduced= np.dot(X_centered,self.components)
        
        return X_reduced
        
    
    def plot_variance_explained(self):
        """
        Creates a bar plot of the variances explained by the principal components.
        """
        if self.explained_variance is not None:
            explained_variance_normalized = self.explained_variance / sum(self.explained_variance) #normalize
            print(explained_variance_normalized)

            num_pcs = len(self.explained_variance)
            x_indices = range(1, num_pcs + 1)

            plt.bar(x_indices, explained_variance_normalized, align='center')
            plt.xlabel('Pincipal component (PC)')
            plt.ylabel('Explained variance normalized')
            plt.title('Explained variance by PC')
            plt.xticks(x_indices,[f'PC{i}' for i in x_indices])
            plt.show()
        else:
            print("The principal components and explained variances have not yet been calculated.")  