def info_gain(left_rows,right_rows,current_uncertainty):
    left,right =left_rows, right_rows

    pleft = len(left)/float(len(left)+len(right))
    pright = len(right)/float(len(left)+len(right))

    ig = current_uncertainty-((pleft*gini(left)) + (pright*gini(right)))
    return ig

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

def checkElement(rows):
    flag = True
    for i in range(len(rows)-1):
        #print rows[i][:-1]
        if rows[i][:-1]!=rows[i+1][:-1]:
            return False
    return True

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

from operator import itemgetter
def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity

def find_best_split(rows):
    best_gain = 0
    best_question = 0
    current_uncertainty = gini(rows)
    n_feature = len(rows[0]) - 1

    for dimensi in range(n_feature):

        data = sorted(rows, key=itemgetter(dimensi))
        #print data
        targetFlag = data[1][-1]
        middleValue = []

        for i in range(len(data) - 1):
            middleValue.append((data[i][dimensi] + data[i + 1][dimensi]) / 2.0)
        for i in range(len(data)):
            if i < len(data) - 1 and data[i][-1] != data[i + 1][-1]:  # kalo berubah
                #print i
                question = Question(dimensi, middleValue[i])
                true_rows, false_rows = partition(data, question)
                ig = info_gain(true_rows, false_rows, current_uncertainty)
                #print "ig", ig
                if ig >= best_gain:
                    best_gain, best_question = ig, question
                targetFlag = data[i][-1]
    #print best_question
    return best_question


def lazyDT(train_data, test_instance):
    if len(class_counts(train_data)) == 1:
        # print "c"
        target = train_data[1][-1]
        return target
    elif checkElement(train_data):
        # print "masuk b"
        cc = class_counts(train_data)
        target = max(cc, key=cc.get)
        return target
    else:

        split = find_best_split(train_data)

        true_rows, false_rows = partition(train_data, split)

        nextData = []
        if split.match(test_instance) == True:
            nextData = true_rows
        else:
            nextData = false_rows
        #print nextData

        return lazyDT(nextData, test_instance)
'''
data = [[1,0.1,1],
        [2,1.1,1],
        [2.5,1.5,1],
        [3.5,0.9,0],
        [3.1,0.8,0]]

test = [2.5,1.5]

a = lazyDT(data,test)
print a
'''
import csv

def read_csv(filename):
    with open(filename) as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

filename = '/home/kkk/LazyDete/trainset.csv'
trainData = read_csv(filename)
filename = '/home/kkk/LazyDete/testset.csv'
testData = read_csv(filename)

print len(testData)
for i in range(len(testData)):
    a = lazyDT(trainData,testData[i][:-1])

    print "predicted: ",a, "seharusnya",testData[i][-1]


