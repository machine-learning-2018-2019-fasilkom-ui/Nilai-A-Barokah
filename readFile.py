import csv


def read_csv(filename):
    with open(filename) as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

filetrain = '/home/kkk/LazyDete/DataSplit/noDuplicate/balance data/randomForest/trainrf2.csv'
trainData = read_csv(filetrain)


