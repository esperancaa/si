from unittest import TestCase
import numpy as np

from Ex6_stratified_train_test_split import stratified_train_test_split
from Exercises.Ex2 import Dataset

class TestStratifiedTrainTestSplit(TestCase):

    def setUp(self):
        """
        Method that will be executed before each test. 
        Here, we create a basic dataset for testing.
        """
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        self.dataset = Dataset(X, y)
        
    def test_stratified_split(self):
        """
        Tests if the stratified split maintains the proportion of classes.
        """
        train_dataset, test_dataset = stratified_train_test_split(self.dataset, test_size=0.33, random_state=42)
        
        train_classes, test_classes = np.unique(train_dataset.y), np.unique(test_dataset.y)
        
        self.assertTrue(np.array_equal(train_classes, np.array([0, 1, 2]))) #0, 1 e 2 devem estar presentes tanto no treino quanto no teste
        self.assertTrue(np.array_equal(test_classes, np.array([0, 1, 2])))
        
        
        train_class_counts = [np.sum(train_dataset.y == c) for c in train_classes] #proporções das classes
        test_class_counts = [np.sum(test_dataset.y == c) for c in test_classes]
        
        # As proporções das classes devem ser semelhantes
        total_samples = len(self.dataset.y)
        train_proportions = [count / total_samples for count in train_class_counts]
        test_proportions = [count / total_samples for count in test_class_counts]
        
        for tp, te in zip(train_proportions, test_proportions):
            self.assertAlmostEqual(tp, te, delta=0.05)  # Permitindo uma pequena variação devido ao tamanho dos dados
        
  
    
