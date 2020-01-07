import unittest
from explore_data import *


class MyTestCase(unittest.TestCase):
    # def test_train_logreg(self):
    #     filename = "./dataset/Train.csv"
    #     train_baseline_model_logreg(filename)

    def test_train_gbrt(self):
        filename = "./dataset/Train.csv"
        train_baseline_model_gbrt(filename)

    def test_train_xgb(self):
        filename = "./dataset/Train.csv"
        train_xgboost(filename)

    # def test_train_model_search(self):
    #     filename = "./dataset/Train.csv"
    #     model_search(filename)
    #
    # def test_train_nn(self):
    #     filename = "./dataset/Train.csv"
    #     train_nn(filename)


if __name__ == '__main__':
    unittest.main()
