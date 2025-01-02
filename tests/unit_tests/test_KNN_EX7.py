import numpy as np

import sys 
sys.path.append("/Users/utilizador/Documents/GitHub/si/src")
from src.si.statistics.euclidean_distance import euclidean_distance
from Exercises.Ex7_KNN import KNNRegressor
from unittest import TestCase
from sklearn.preprocessing import StandardScaler

from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split

class TestKNNRegressor(TestCase):

    def setUp(self):
        
       self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

       self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        
        knn = KNNRegressor(k=3, distance=None)

        knn._fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        
        knn = KNNRegressor(k=1, distance=euclidean_distance)
        train_dataset, test_dataset = train_test_split(self.dataset)

        knn._fit(train_dataset)
        
        predictions = knn._predict(test_dataset.features)

        
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        accuracy = np.mean(predictions == test_dataset.y)
        self.assertGreater(accuracy, 0.8)  # Exige pelo menos 80% de precisão

    def test_score(self):
        
        knn = KNNRegressor(k=3, distance=euclidean_distance)
        train_dataset, test_dataset = train_test_split(self.dataset)
        knn._fit(train_dataset)
        score = knn._score(test_dataset)
        self.assertLess(score, 100)