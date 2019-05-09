import pandas as pd
import time
from C45 import C45
from  TreeBuilder import TreeBuilder
import numpy as np

def fit(training_path, tree_path):
    # read data training
    df=pd.read_csv(training_path, sep=',',header=None)

    # remove index column
    XY = df.values
    XXY = XY[:,1:]

    ##
    # Fitting process
    # 2nd param, 5 means select a feature as discrete if unique value of the feature is less than equal 5, else set feature as numeric feature
    # 3rd param, 6 means we use 6 pararel process at a time when calculating information gain (in SplitAttribute class)
    ##
    feature_threshold = 100
    pool_size = 8
    tree = C45(XXY, feature_threshold, pool_size)
    tree.fit()

    df2 = pd.DataFrame(tree.tree)
    df2.to_csv(tree_path)


def buildTree(matriks_path):
    ##
    # Build tree
    ##

    #matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/result_training_set4v2.csv'
    df3 = pd.read_csv(matriks_path, sep=',',header=None)
    df3 = df3.where((pd.notnull(df3)), None)
    matriks = df3.values
    matriks = matriks[1:,1:]

    tree_builder = TreeBuilder(matriks, [1,0])
    tree_builder.build()
    print(tree_builder.confusion_matrix)
    return tree_builder

def predict(tree_path, test_data_path, prediction_path):
    tree = buildTree(tree_path)

    test_data = pd.read_csv(test_data_path, sep=',', header=None)
    test_data = test_data.values

    prediction, confusion_mat = tree.predict(test_data, False)
    print(confusion_mat)
    pd.DataFrame(prediction).to_csv(prediction_path)


def predictWithEnsemble():
    testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/DataSplit/testing_set1.csv'
    df4 = pd.read_csv(testing_path, sep=',', header=None)
    matriks_testing = df4.values

    testing_result_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result.csv'
    df5 = pd.read_csv(testing_result_path, sep=',', header=None)
    matriks_testing_result = df5.values
    matriks_testing_result = matriks_testing_result[1:, 1:].astype(np.float64)

    final_result = []
    y = matriks_testing[1:, -1].astype(np.float64)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    idx = 0
    for data in matriks_testing_result:
        actual_class = y[idx]
        vote_val = (data[1] + data[2] + data[3] + data[0]) / 4
        if (vote_val > 0.5):
            predicted_class = 1
        else:
            predicted_class = 0

        if (predicted_class == actual_class):
            if (actual_class == 1):
                TP += 1
            else:
                TN += 1
        else:
            if (actual_class == 1):
                FN += 1
            else:
                FP += 1


        idx += 1
        final_result.append(predicted_class)

    print("Final Accuracy: "+str((TP+TN)/(TP+FP+TN+FN)))
    print("Final Precision: " + str(TP / (TP + FP )))
    print("Final Recall: " + str(TP / (TP + FN)))


# training
timestamp = str(time.time())
training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/randomForest/trainrf3.csv'
tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/' + timestamp + '-tree.csv'
fit(training_path, tree_path)

# timestamp = str(time.time())
# training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/randomForest/trainrf2.csv'
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/' + timestamp + '-tree.csv'
# fit(training_path, tree_path)
#
# timestamp = str(time.time())
# training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/randomForest/trainrf3.csv'
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/' + timestamp + '-tree.csv'
# fit(training_path, tree_path)
#
# timestamp = str(time.time())
# training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/randomForest/trainrf4.csv'
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/' + timestamp + '-tree.csv'
# fit(training_path, tree_path)
#
# timestamp = str(time.time())
# training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/randomForest/trainrf5.csv'
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/' + timestamp + '-tree.csv'
# fit(training_path, tree_path)


# testing
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/tree-1.csv'
# testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/balancedDataTest.csv'
# prediction_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result-eager-balance-1.csv'
# predict(tree_path, testing_path, prediction_path)
#
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/tree-2.csv'
# testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/balancedDataTest.csv'
# prediction_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result-eager-balance-2.csv'
# predict(tree_path, testing_path, prediction_path)
#
#
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/tree-3.csv'
# testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/balancedDataTest.csv'
# prediction_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result-eager-balance-3.csv'
# predict(tree_path, testing_path, prediction_path)
#
#
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/tree-4.csv'
# testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/balancedDataTest.csv'
# prediction_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result-eager-balance-4.csv'
# predict(tree_path, testing_path, prediction_path)
#
#
# tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/tree-5.csv'
# testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/balancedData/balancedDataTest.csv'
# prediction_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result-eager-balance-5.csv'
# predict(tree_path, testing_path, prediction_path)

