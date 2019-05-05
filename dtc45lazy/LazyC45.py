import time
import numpy as np
from AttributeSelector import AttributeSelector


class LazyC45:
    def __init__(self, data, threshold=100, pool_size=5):
        self.pool_size = pool_size
        self.tree = []
        self.cur_num_node = 0

        self.train_data = data
        self.classes = []
        self.attributes_info = {}

        self.constructClass()
        self.constructFeatureDesc(threshold)

    def constructClass(self):
        target = np.unique(self.train_data[1:, -1].astype(np.float64), return_counts=False)
        self.classes = target

    def constructFeatureDesc(self, threshold):
        ##
        # data_type -> nominal/discrete attribute = 0 ; numeric/continues attribute = 1;
        #
        for i in range(self.train_data.shape[1]):
            single_desc = {'data_type': 0}

            # use this function if a feature has double value
            # unique, counts = np.unique(self.dataset[1:, i].astype(np.float64), return_counts=True)

            # use this function if you are sure there is no feature containing double value
            # unique, counts = np.unique(self.dataset[1:, i], return_counts=True)

            try:
                unique, counts = np.unique(self.train_data[1:, i].astype(np.float64), return_counts=True)
            except:
                unique, counts = np.unique(self.train_data[1:, i], return_counts=True)

            single_desc['name'] = self.train_data[0, i]
            if len(unique) > threshold:
                single_desc['data_type'] = 1
            single_desc['unique_edge'] = unique
            single_desc['unique_counts'] = counts
            single_desc['col_idx'] = i
            self.attributes_info[self.train_data[0, i]] = single_desc

    def predict(self, test_data):
        start = time.time()
        print("...... Start predict " + str(time.ctime(int(start))) + "...... \n")

        result = self.predictInternal(self.train_data, test_data)

        print("\n\n\n")
        print("start time :" + str(time.ctime(int(start))) + "\n")
        print("finish time :" + str(time.ctime(int(time.time()))) + "\n")
        print("Fitting done in  " + str(time.time() - start))

        return result

    def predictInternal(self, cur_train_data, cur_test_data):
        start = time.time()
        predicted = np.empty((0,2))

        ###### Check terminate condition
        # 1. when dataset is empty
        if (cur_train_data.shape[0] == 1):
            print("Got empty train data\n")
            return []

        cur_train_class, cur_train_class_count = np.unique(cur_train_data[1:, -1].astype(np.float64), return_counts=True)

        # 2. when target is already homogen
        #    or
        # 3. when no target or attribute
        if (len(cur_train_class) == 1 or cur_train_data.shape[1] < 2):
            max_count, selected_class = self.selectMajorityClass(cur_train_class, cur_train_class_count)

            predicted = np.zeros((cur_test_data.shape[0], 2))
            predicted[:, 0] = cur_test_data[:, 0]
            predicted[:, 1] = np.full((1,),selected_class)
            return predicted

        ###### Check terminate condition DONE



        ## select a feature as current node
        splitter = AttributeSelector(self.pool_size, self.attributes_info, cur_train_data, cur_train_class)
        splitter.doSelect()

        cur_attr_name = splitter.selected_attr_name
        cur_attr_idx = splitter.selected_attr_idx
        cur_selected_spillter = splitter.selected_splitter
        cur_gain = splitter.selected_gain
        cur_split_info = splitter.selected_split_info

        if (cur_selected_spillter != None and len(cur_selected_spillter) == 1): # if feature is numeric feature. Assuming feature that has one unique edge is numeric feature.
            no_header_data = cur_train_data[1:, :]
            cur_selected_spillter = cur_selected_spillter[0]

            # 1. recursively predict test data with value less than equal selected splitter
            test_data_less = cur_test_data[np.where(cur_test_data[:, cur_attr_idx+1].astype(np.float64) <= cur_selected_spillter.astype(np.float64))]

            if (len(test_data_less) > 0):
                train_data_less = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) <= cur_selected_spillter.astype(np.float64))]
                train_data_less = np.concatenate(([cur_train_data[0, :]], train_data_less), axis=0)
                train_data_less = np.delete(train_data_less, cur_attr_idx, 1)
                test_data_less = np.delete(test_data_less, cur_attr_idx+1, 1)

                left_child_predicted = self.predictInternal(train_data_less, test_data_less)
                if (left_child_predicted.shape[0] > 0):
                    predicted = np.append(predicted, left_child_predicted).reshape(left_child_predicted.shape[0], left_child_predicted.shape[1])
                print("left child fit done. parent node: " + str(cur_attr_name)+"\n")


            # 2. recursively predict test data with value greater than selected splitter
            test_data_greater = cur_test_data[np.where(cur_test_data[:, cur_attr_idx+1].astype(np.float64) > cur_selected_spillter.astype(np.float64))]

            if (len(test_data_greater) > 0):
                train_data_greater = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) > cur_selected_spillter.astype(np.float64))]
                train_data_greater = np.concatenate(([cur_train_data[0, :]], train_data_greater), axis=0)
                train_data_greater = np.delete(train_data_greater, cur_attr_idx, 1)
                test_data_greater = np.delete(test_data_greater, cur_attr_idx+1, 1)

                right_child_predicted = self.predictInternal(train_data_greater, test_data_greater)
                if (right_child_predicted.shape[0] > 0):
                    predicted = np.append(predicted, right_child_predicted).reshape(predicted.shape[0]+right_child_predicted.shape[0], right_child_predicted.shape[1])
                print("right child fit done. parent node: " + str(cur_attr_name)+"\n")

        elif (cur_selected_spillter != None): # if feature is discrete feature. Assuming feature that has more than one unique edge is discrete feature.
            no_header_data = cur_train_data[1:, :]
            for i in range(len(cur_selected_spillter)):
                selected_test_data = cur_test_data[np.where(cur_test_data[:, cur_attr_idx+1] == cur_selected_spillter[i])]

                if(len(selected_test_data) > 0 ):
                    selected_train_data = no_header_data[np.where(no_header_data[:, cur_attr_idx] == cur_selected_spillter[i])]
                    selected_train_data = np.concatenate(([cur_train_data[0, :]], selected_train_data), axis=0)
                    selected_train_data = np.delete(selected_train_data, cur_attr_idx, 1)
                    selected_test_data = np.delete(selected_test_data, cur_attr_idx+1, 1)

                    sub_result = self.predictInternal(selected_train_data, selected_test_data)
                    if(sub_result.shape[0] > 0 ):
                        predicted = np.append(predicted, sub_result).reshape(predicted.shape[0]+sub_result.shape[0], sub_result.shape[1])
        else:
            max_count, selected_class = self.selectMajorityClass(cur_train_class, cur_train_class_count)

            predicted = np.zeros((cur_test_data.shape[0], 2))
            predicted[:, 0] = cur_test_data[:, 0]
            predicted[:, 1] = np.full((1,), selected_class)
            return predicted

        return predicted

    def selectMajorityClass(self, curr_target, curr_target_count):
        max_count, selected_class = 0, 0
        for i in range(len(curr_target)):
            if (curr_target_count[i] > max_count):
                max_count = curr_target_count[i]
                selected_class = curr_target[i]
        return max_count, selected_class









