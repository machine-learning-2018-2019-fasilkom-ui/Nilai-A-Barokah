import time
import numpy as np
from AttributeSelector import AttributeSelector


class C45v2:
    def __init__(self, data, threshold, pool_size):
        self.pool_size = pool_size
        self.tree = []
        self.cur_num_node = 0

        self.dataset = data
        self.classes = []
        self.attributes_info = {}

        self.constructClass()
        self.constructFeatureDesc(threshold)

    def constructClass(self):
        target = np.unique(self.dataset[1:, -1].astype(np.float64), return_counts=False)
        self.classes = target

    def constructFeatureDesc(self, threshold):
        ##
        # data_type -> nominal/discrete attribute = 0 ; numeric/continues attribute = 1;
        #
        for i in range(self.dataset.shape[1]):
            single_desc = {'data_type': 0}

            # use this function if a feature has double value
            # unique, counts = np.unique(self.dataset[1:, i].astype(np.float64), return_counts=True)

            # use this function if you are sure there is no feature containing double value
            # unique, counts = np.unique(self.dataset[1:, i], return_counts=True)

            try:
                unique, counts = np.unique(self.dataset[1:, i].astype(np.float64), return_counts=True)
            except:
                unique, counts = np.unique(self.dataset[1:, i], return_counts=True)

            single_desc['name'] = self.dataset[0, i]
            if len(unique) > threshold:
                single_desc['data_type'] = 1
            single_desc['unique_edge'] = unique
            single_desc['unique_counts'] = counts
            single_desc['col_idx'] = i
            self.attributes_info[self.dataset[0, i]] = single_desc

    def fit(self):
        start = time.time()
        print("...... Start fitting " + str(time.ctime(int(start))) + "...... \n")

        self.fitInternal(self.dataset, 0, None, None)

        print("\n\n\n")
        print("start time :" + str(time.ctime(int(start))) + "\n")
        print("finish time :" + str(time.ctime(int(time.time()))) + "\n")
        print("Fitting done in  " + str(time.time() - start))

    def fitInternal(self, currentData, parent_id, parent_edge, parent_data_type):
        print("...... Start Fitting " + str(self.cur_num_node) + " nd node, parent id : " + str(parent_id) + ", parent edge :" + str(parent_edge) + " ...... \n")
        start = time.time()

        ###### Check terminate condition
        # 1. when dataset is empty
        if (currentData.shape[0] == 1):
            print("......  Fitting " + str(self.cur_num_node) + " nd node, parent id : " + str(parent_id) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            print("Got empty dataset\n")
            return None

        curr_target, curr_target_count = np.unique(currentData[1:, -1].astype(np.float64), return_counts=True)
        targetCompostion = None #  composition number of unique target. we need this to calculate error training
        for count in curr_target_count:
            if (targetCompostion != None):
                targetCompostion += "-" + str(count)
            else:
                targetCompostion = str(count)

        self.cur_num_node += 1
        cur_idx = self.cur_num_node

        # 2. when target is already homogen
        if (len(curr_target) == 1):
            self.tree.append([cur_idx, parent_id, None, None, 0.0, 0.0, parent_edge, 0.0, curr_target[0], targetCompostion, None, parent_data_type])
            print("......  Fitting " + str(self.cur_num_node) + " nd node, parent id : " + str(parent_id) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            return

        # 3. when no target or attribute
        if (currentData.shape[1] < 2):
            max_count, selected_class = self.selectMajorityClass(curr_target, curr_target_count)
            self.tree.append([cur_idx, parent_id, None, None, 0.0, 0.0, parent_edge, 0.0, selected_class, targetCompostion, None, parent_data_type])
            print("......  Fitting " + str(self.cur_num_node) + " nd node, parent id : " + str(parent_id) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            return
        ###### Check terminate condition

        ## select a feature as current node
        splitter = AttributeSelector(self.pool_size, self.attributes_info, currentData, curr_target)
        splitter.doSelect()

        cur_attr_name = splitter.selected_attr_name
        cur_attr_idx = splitter.selected_attr_idx
        cur_threshold = splitter.selected_splitter
        cur_gain = splitter.selected_gain
        cur_split_info = splitter.selected_split_info
        thresholdCompostion = None # composition number of unique edge of selected/current node. we need this to calculate error training

        if (cur_threshold != None and len(cur_threshold) == 1): # if feature is numeric feature. Assuming feature that has one unique edge is numeric feature.
            cur_feature_type = 'numeric'
            no_header_data = currentData[1:, :]
            cur_threshold = cur_threshold[0]

            # 1. recursively fit left child
            less_eq_data = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) <= cur_threshold.astype(np.float64))]
            less_eq_data = np.concatenate(([currentData[0, :]], less_eq_data), axis=0)
            XXY_new2 = np.delete(less_eq_data, cur_attr_idx, 1)
            thresholdCompostion = str(XXY_new2.shape[0]-1)
            self.fitInternal(XXY_new2, cur_idx, '<=' + str(cur_threshold), cur_feature_type)
            print("left child fit done. parent node: " + str(cur_attr_name)+"\n")


            # 2. recursively fit right child
            greater_data = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) > cur_threshold.astype(np.float64))]
            greater_data = np.concatenate(([currentData[0, :]], greater_data), axis=0)
            XXY_new2 = np.delete(greater_data, cur_attr_idx, 1)
            thresholdCompostion += "-" + str(XXY_new2.shape[0]-1)
            self.fitInternal(XXY_new2, cur_idx, '>' + str(cur_threshold), cur_feature_type)
            print("right child fit done. parent node: " + str(cur_attr_name)+"\n")

        elif (cur_threshold != None): # if feature is discrete feature. Assuming feature that has more than one unique edge is discrete feature.
            cur_feature_type = 'nominal'
            no_header_data = currentData[1:, :]
            cur_threshold_str = None
            for i in range(len(cur_threshold)):
                cur_threshold_str = str(cur_threshold[i]) if cur_threshold_str == None else cur_threshold_str +"-" + str(cur_threshold[i])
                selected_dataset = no_header_data[np.where(no_header_data[:, cur_attr_idx] == cur_threshold[i])]
                selected_dataset = np.concatenate(([currentData[0, :]], selected_dataset), axis=0)
                selected_dataset = np.delete(selected_dataset, cur_attr_idx, 1)
                thresholdCompostion = str(selected_dataset.shape[0]-1) if thresholdCompostion == None else thresholdCompostion+"-"+str(selected_dataset.shape[0]-1)
                self.fitInternal(selected_dataset, cur_idx, cur_threshold[i], cur_feature_type)
            cur_threshold = cur_threshold_str
            print("left child fit done. parent node: " + str(cur_attr_name) + "\n")
        else:
            print("cur idx: "+str(cur_idx))
            print(currentData)
            max_count, selected_class = self.selectMajorityClass(curr_target, curr_target_count)
            self.tree.append(
                [cur_idx, parent_id, None, None, 0.0, 0.0, parent_edge, 0.0, selected_class, targetCompostion, None,
                 parent_data_type])
            return


        # save tree to matrix. format : [treeIdx, parentId, attrName, colIndex, gain, splitInfo, parentEdge, threshold, leaveVal, targetCompostion, thresholdCompostion, cur_feature_type]
        col_idx = None if cur_attr_name == None else self.attributes_info[cur_attr_name]['col_idx']
        self.tree.append(
            [cur_idx, parent_id, cur_attr_name, col_idx, cur_gain,
             cur_split_info, parent_edge, cur_threshold, None, targetCompostion, thresholdCompostion, cur_feature_type])

        print("......  Fitting " + str(self.cur_num_node) + " nd node, parent id : " + str(parent_id) + ", parent edge :" + str(parent_edge) + " ...... Done in " + str(time.time() - start) + "!!!\n")

    def selectMajorityClass(self, curr_target, curr_target_count):
        max_count, selected_class = 0, 0
        for i in range(len(curr_target)):
            if (curr_target_count[i] > max_count):
                max_count = curr_target_count[i]
                selected_class = curr_target[i]
        return max_count, selected_class









