import time
import numpy as np
from Node import Node

class TreeBuilder:
    def __init__(self, matrix, classes):
        self.matrix = matrix
        self.tree = None
        self.prunedTree = None
        self.classes = classes

        self.confusion_matrix = {}

    def build(self):
        # init confusion matrix
        self.confusion_matrix = self.initConfusionMat(self.confusion_matrix, self.classes)
        self.tree = self.buildInternal(0, None)

    def buildInternal(self, cur_parent_idx, cur_parent_edge):
        cur_matriks = self.matrix[np.where( (self.matrix[:,1] == cur_parent_idx) and (self.matrix[:,6] == cur_parent_edge))]
        child_nodes = {}

        root_mat = cur_matriks[0]

        idx = root_mat[0]
        name = root_mat[2]
        edges_string = root_mat[7]
        leaf_val = root_mat[8]
        class_composition_string = root_mat[9]
        edge_composition_string = root_mat[10]
        feature_type = root_mat[11]

        # if current node is leave
        if leaf_val != None and leaf_val != "":
            current_node = Node(name, None, cur_parent_edge, None, class_composition_string,
                                None, feature_type)

            # count correct classification for training data
            self.confusion_matrix[str(leaf_val) + "_true"] += current_node.class_composition[0]

            # count mismatch classification for training data
            if (len(current_node.class_composition) > 1) :
                self.confusion_matrix[str(leaf_val) + "_false"] += current_node.class_composition[1]
        else :
            current_node = Node(name, None, cur_parent_edge, edges_string, class_composition_string,
                                edge_composition_string, feature_type)
            if(current_node.feature_type == 'numeric'):
                splitter_point = current_node.edges[0]

                # build left child tree
                child_nodes['<='+str(splitter_point)] = self.buildInternal(idx, '<='+str(splitter_point))

                # build right child tree
                child_nodes['>'+str(splitter_point)] = self.buildInternal(idx, '>'+str(splitter_point))

                current_node.childs = child_nodes
            else:
                for i in range(len(current_node.edges)):
                    child_nodes[current_node.edges[i]] = self.buildInternal(idx, current_node.edges[i])
                current_node.childs = child_nodes

        return current_node

    def getAccuracy(self):
        return 0.0

    def getPrecision(self):
        return 0.0

    def getRecall(self):
        return 0.0

    def predict(self, test_data, tree_type, is_using_prune_tree):
        start_time = time.time()
        self.printStartTimestamp(start_time)

        # init confusion matrix
        confusion_mat = self.initConfusionMat({}, self.classes)

        for i in range(test_data.shape[0]):
            row = test_data[i,:-1]
            actual_class = test_data[i,-1]
            predicted_class = self.predictInternal(row, is_using_prune_tree)

            if( predicted_class ==  actual_class):
                confusion_mat["true"][str(actual_class)] +=1
            else:
                confusion_mat["false"][str(actual_class)] +=1

        self.printFinishTimestamp(start_time)
        return confusion_mat

    def predictInternal(self, is_using_prune_tree):
        tree = self.prunedTree if is_using_prune_tree else self.tree

    def initConfusionMat(self, conf_mat, classes):
        for i in range(len(classes)):
            conf_mat["true"][str(classes[i])] = 0
            conf_mat["false"][str(classes[i])] = 0
        return conf_mat

    def prune(self):
        start_time = time.time()
        self.printFinishTimestamp(start_time)

    ##
    # Logger
    #
    def printStartTimestamp(self, start_time):
        print("...... Start Predict " + str(time.ctime(int(start_time))) + " ...... \n")

    def printFinishTimestamp(self, start_time):
        print("start time :" + str(time.ctime(int(start_time))) + "\n")
        print("finish time :" + str(time.ctime(int(time.time()))) + "\n")
        print("Predict done in  " + str(time.time() - start_time))
