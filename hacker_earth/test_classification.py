import unittest
from explore_data import *


class MyTestCase(unittest.TestCase):
    def test_classify(self):
        filename = "./dataset/Test.csv"
        load_model()
        classify(filename)

    # def test_classify_xgb(self):
    #     filename = "./dataset/Test.csv"
    #     load_model()
    #     classify_xgb(filename)

    # def test_classify_nn(self):
    #     filename = "./dataset/Test.csv"
    #     load_model_nn()
    #     classify_nn(filename)


if __name__ == '__main__':
    unittest.main()
