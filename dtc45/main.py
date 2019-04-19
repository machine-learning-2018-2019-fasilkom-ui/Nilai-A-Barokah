import pandas as pd
import time
from C45v2 import C45v2
from  TreeBuilder import TreeBuilder
import numpy as np

def fit():
    # read data training
    training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/DataSplit/training_set4.csv'
    df=pd.read_csv(training_path, sep=',',header=None)

    # remove index column
    XY = df.values
    XXY = XY[:,:]

    ##
    # Fitting process
    # 2nd param, 5 means select a feature as discrete if unique value of the feature is less than equal 5, else set feature as numeric feature
    # 3rd param, 6 means we use 6 pararel process at a time when calculating information gain (in SplitAttribute class)
    ##
    tree = C45v2(XXY, 100, 6)
    tree.fit()

    tree_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/'+str(time.time())+'-tree.csv'
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

def predict():
    matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/result_training_set1v2.csv'
    tree1 = buildTree(matriks_path)

    matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/result_training_set2v2.csv'
    tree2 = buildTree(matriks_path)

    matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/result_training_set3v2.csv'
    tree3 = buildTree(matriks_path)

    matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/result_training_set4v2.csv'
    tree4 = buildTree(matriks_path)

    matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/result_training_set5v3.csv'
    tree5 = buildTree(matriks_path)

    testing_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/DataSplit/testing_set1.csv'
    df4 = pd.read_csv(testing_path, sep=',', header=None)
    matriks_testing = df4.values

    #result1, confusion_mat1 = tree1.predict(matriks_testing, False)
    result2, confusion_mat2 = tree2.predict(matriks_testing, False)
    result3, confusion_mat3 = tree3.predict(matriks_testing, False)
    result4, confusion_mat4 = tree4.predict(matriks_testing, False)
    result5, confusion_mat5 = tree5.predict(matriks_testing, False)

    print(confusion_mat2)
    print(confusion_mat3)
    print(confusion_mat4)
    print(confusion_mat5)

    all_result_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/test-result.csv'
    df5 = pd.DataFrame()
    #df5['r1'] = result1
    df5['r2'] = result2
    df5['r3'] = result3
    df5['r4'] = result4
    df5['r5'] = result5
    df5.to_csv(all_result_path)


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

predictWithEnsemble()