import time
import numpy as np
import pandas as pd
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
        self.tree = self.buildInternal(0, None, None)

    def buildInternal(self, parent_idx, parent_edge, parent_edge_operator):
        # parent_edge = parent_edge if parent_edge == None else parent_edge.astype(np.float64)
        if parent_edge == None :
            cur_matriks = self.matrix[np.where((self.matrix[:,1] == parent_idx) & (self.matrix[:, 6] == parent_edge) & (self.matrix[:, 7] == parent_edge_operator))]
        else :
            temp_matriks = self.matrix[np.where( (self.matrix[:, 1] == parent_idx) & (self.matrix[:, 6] != None) ) ]
            parent_edge = np.float(parent_edge)
            cur_matriks = temp_matriks[np.where( (temp_matriks[:, 6] == parent_edge) & (temp_matriks[:, 7] == parent_edge_operator) )]
        child_nodes = {}

        root_mat = cur_matriks[0]

        idx = root_mat[0]
        name = root_mat[2]
        edges_string = root_mat[8]
        leaf_val = root_mat[9]
        class_composition_string = root_mat[10]
        edge_composition_string = root_mat[11]
        feature_type = root_mat[12]

        # if current node is leave
        if leaf_val != None and leaf_val != "":
            current_node = Node(name, None, parent_edge, None, class_composition_string,
                                None, feature_type, leaf_val)

            # count mismatch classification for training data
            if (len(current_node.class_composition) > 1) :
                first_composition = int(current_node.class_composition[0])
                second_composition = int(current_node.class_composition[1])
                self.confusion_matrix['true'][leaf_val] += first_composition if first_composition >= second_composition else second_composition
                self.confusion_matrix['false'][leaf_val] += second_composition if second_composition < first_composition else first_composition
            else:
                # count correct classification for training data
                self.confusion_matrix['true'][leaf_val] += int(current_node.class_composition[0])

        else :
            current_node = Node(name, None, parent_edge, edges_string, class_composition_string,
                                edge_composition_string, feature_type, None)
            if(current_node.feature_type == 'numeric'):
                splitter_point = current_node.edges[0]

                # build left child tree
                child_nodes['<='] = self.buildInternal(idx, splitter_point, '<=')

                # build right child tree
                child_nodes['>'] = self.buildInternal(idx, splitter_point, '>')

                current_node.childs = child_nodes
            else:
                for i in range(len(current_node.edges)):
                    child_nodes[current_node.edges[i]] = self.buildInternal(idx, current_node.edges[i], None)
                current_node.childs = child_nodes

        return current_node

    def getAccuracy(self):
        true_val = 0
        false_val = 0

        for val in self.confusion_matrix['true']:
            true_val += self.confusion_matrix['true'][val]

        for val in self.confusion_matrix['false']:
            false_val += self.confusion_matrix['false'][val]

        return true_val / (true_val+false_val)

    def getPrecision(self):
        return self.confusion_matrix['true'][1] / (self.confusion_matrix['true'][1] + self.confusion_matrix['false'][0])

    def getRecall(self):
        return self.confusion_matrix['true'][1] / (self.confusion_matrix['true'][1] + self.confusion_matrix['false'][1])

    def predict(self, test_data, is_using_prune_tree):
        start_time = time.time()
        self.printStartTimestamp(start_time)

        # init confusion matrix
        confusion_mat = self.initConfusionMat({}, self.classes)
        result = []

        header = test_data[0, :-1]
        row_data = test_data[1:,:]
        for i in range(row_data.shape[0]):
            row = row_data[i,:-1]
            single_test_data = pd.DataFrame([row], columns = header)
            actual_class = int(float(row_data[i,-1]))
            predicted_class = int(float(self.predictInternal(single_test_data, is_using_prune_tree)))

            result.append(predicted_class);

            if( predicted_class ==  actual_class):
                confusion_mat['true'][actual_class] +=1
            else:
                confusion_mat['false'][actual_class] +=1
            #print("Done fitting data row : "+str(i) + " " + str(time.ctime(int(time.time()))))

        self.printFinishTimestamp(start_time)
        return result, confusion_mat

    def predictInternal(self, row, is_using_prune_tree):
        tree = self.prunedTree if is_using_prune_tree else self.tree
        return self.traceTree (row, tree)

    def traceTree(self, row, tree):
        if ( ( tree.childs == None or len(tree.childs) == 0)  and tree.leave_value != None):
            return tree.leave_value
        elif (tree.feature_type == 'numeric'):
            split_point = float(tree.edges[0])
            cur_feature_val = float(row[tree.name][0])

            if (cur_feature_val <= split_point):
                return self.traceTree(row, tree.childs['<='])
            else:
                return self.traceTree(row, tree.childs['>'])
        elif (tree.feature_type == 'nominal'):
            print("TODO Implementation!")


    def initConfusionMat(self, conf_mat, classes):
        conf_mat["true"] = {}
        conf_mat["false"] = {}
        for i in range(len(classes)):
            conf_mat["true"][classes[i]] = 0
            conf_mat["false"][classes[i]] = 0
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
