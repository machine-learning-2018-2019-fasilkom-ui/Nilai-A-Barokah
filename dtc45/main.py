import pandas as pd
import time
from C45v2 import C45v2
from  TreeBuilder import TreeBuilder

def fit():
    # read data training
    training_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/datasets/DataSplit/training_set5.csv'
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


def buildTree():
    ##
    # Build tree
    ##

    matriks_path = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/matriks.csv'
    df3 = pd.read_csv(matriks_path, sep=',',header=None)
    matriks = df3.values
    matriks = matriks[1:,1:]

    tree_builder = TreeBuilder(matriks, [1,0])
    tree_builder.build()

def predict(tree):
    ##
    # Predict
    ##

    testing_path = ''

fit()
