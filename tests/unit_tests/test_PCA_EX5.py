from unittest import TestCase
import numpy as np
from si.data.dataset import Dataset
from Exercises.Ex5_PCA import PCA 

class TestPCA(TestCase):

    def setUp(self):
        
        self.X = np.array([[2.5, 2.4],
                           [0.5, 0.7],
                           [2.2, 2.9],
                           [1.9, 2.2],
                           [3.1, 3.0],
                           [2.3, 2.7],
                           [2.0, 1.6],
                           [1.0, 1.1],
                           [1.5, 1.6],
                           [1.1, 0.9]])
        self.dataset = Dataset(X=self.X, y=None)
        self.pca = PCA(n_components=1)

    def test_initialization(self):
        
        self.assertEqual(self.pca.n_components, 1)
        self.assertIsNone(self.pca.components)
        self.assertIsNone(self.pca.mean)
        self.assertIsNone(self.pca.explained_variance)

    def test_fit(self):
        
        self.pca.fit(self.dataset)
        self.assertIsNotNone(self.pca.components)
        self.assertIsNotNone(self.pca.mean)
        self.assertIsNotNone(self.pca.explained_variance)

        
        self.assertEqual(self.pca.mean.shape, (self.X.shape[1],)) #média tem o tamanho correto

        
        self.assertEqual(self.pca.components.shape, (self.X.shape[1], self.pca.n_components)) #componentes principais têm o tamanho esperado


    def test_explained_variance(self):
        self.pca.fit(self.dataset)
        explained_variance = self.pca.explained_variance

    
        self.assertLessEqual(np.sum(explained_variance), 1) #menor ou igual a 1

