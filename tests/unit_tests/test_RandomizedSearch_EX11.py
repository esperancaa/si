from unittest import TestCase
from si.models.logistic_regression import LogisticRegression
from Exercises.Ex11_randomized_search import randomized_search_cv
from Exercises.Ex2 import Dataset
import numpy as np

class TestRandomizedSearchCV(TestCase):
    def setUp(self):
        # Dataset sintético
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 2, 100)
        self.dataset = Dataset(self.X, self.y)
        
        # Modelo e grid de hiperparâmetros
        self.model = LogisticRegression()
        self.hyperparameter_grid = {
            'l2_penalty': [0.1, 1, 10],
            'alpha': [0.001, 0.01, 0.1],
        }

    def test_randomized_search_cv(self):
        # Executa a busca aleatória
        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.hyperparameter_grid,
            cv=3,
            n_iter=5
        )
        
        # Verifica se os resultados são válidos
        self.assertIn('scores', results)
        self.assertIn('hyperparameters', results)
        self.assertIn('best_hyperparameters', results)
        self.assertIn('best_score', results)
        self.assertEqual(len(results['scores']), 5)
        self.assertEqual(len(results['hyperparameters']), 5)