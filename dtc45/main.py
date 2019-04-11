#import Node
#import Branch
#import Label
#import BranchType

# init data model
#l1 = Label("yes", 0.1)
#b1 = Branch(BranchType.dischretes, "sunny", None, None, False)
#n1 = Node(None, None, True, b1, 0.0, 0.0, [])

# for feature in features_name:
#   for label in labels_name:

# foreach X_label
# init sorted dataset foreach label with doc index
# calculate gain ratio for each fatures
# calculate Entropy(label)
# pi ; {pi(yes) = number of yes / number of data, pi(no) = number of no / number of data}
# Entropy(T) = E(T) sum(-pi*2log(pi))
# double maxGainValue, string maxGainLabel
# select only collumn X_label_i and y_label
# calculate entropy E(T, X) = sum(p(c)*E(c)) = sum(P(sunny)*E(sunny, yes, no)+ P(overcast)*E(overcast, yes, no))
# calculate gain = Gain(T,X) = Entropy(T) - Entropy(T,X)
# set feature with highest gr as current node
# split dataset XY respectively with the unselected features
# recursively do fit foreach spillted XY or subtree


import pandas as pd
from C45v2back import C45v2back
import time

filePath = '/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/training_2.csv'
df=pd.read_csv(filePath, sep=',',header=None)
XY = df.values
XXY = XY[:,1:]
tree = C45v2back(XXY, 100)
tree.fit()

tree.tree.to_csv(r'/home/adeliaherlisa/Documents/s2/2018 genap/machine learning/TK Akhir/result/'+str(time.time())+'-tree.csv')