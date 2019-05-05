import pandas as pd
import time
from LazyC45 import LazyC45

def predict():
    # read data training
    train_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/DataSplit/training_set4.csv'
    train_data =pd.read_csv(train_path, sep=',',header=None)
    train_data = train_data.values

    # read data testing
    test_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/DataSplit/testing_set1.csv'
    test_data = pd.read_csv(test_path, sep=',', header=None)
    test_data = test_data.values
    test_data = test_data[1:,:-1]

    ##
    # predict process
    # 2nd param, 5 means select a feature as discrete if unique value of the feature is less than equal 5, else set feature as numeric feature
    # 3rd param, 6 means we use 6 pararel process at a time when calculating information gain (in SplitAttribute class)
    ##
    feature_threshold = 100
    pool_size = 6
    tree = LazyC45(train_data, feature_threshold, pool_size)

    predicted = tree.predict(test_data)
    predicted_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/' + str(
        time.time()) + '-predicted.csv'

    pd.DataFrame(predicted).to_csv(predicted_path)

predict()
