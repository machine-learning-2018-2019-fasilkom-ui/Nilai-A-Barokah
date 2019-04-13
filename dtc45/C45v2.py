import time
import numpy as np
from SplitAttribute import SplitAttribute


class C45v2:
    def __init__(self, data, threshold):
        self.tree = []
        self.currNumNode = 0

        self.dataset = data
        self.classes = []
        self.attributes_info = {}

        self.constructClass()
        self.constructFeatureDesc(threshold)

    def constructClass(self):
        target = np.unique(self.dataset[1:, -1].astype(np.float64), return_counts=False)
        self.classes = target

    def constructFeatureDesc(self, threshold):
        # "data_type" -> nominal/discrete attribute = 0 ; numeric/continues attribute = 1;
        for i in range(self.dataset.shape[1]):
            single_desc = {'data_type': 0}
            #unique, counts = np.unique(self.dataset[1:, i].astype(np.float64), return_counts=True)
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

        self.fitInternal(self.dataset, 0, None)

        print("\n\n\n")
        print("current time :" + str(time.ctime(int(start))) + "\n")
        print("current time :" + str(time.ctime(int(time.time()))) + "\n")
        print("Fitting done in  " + str(time.time() - start))

    def fitInternal(self, currentData, parentId, parent_edge):
        print("...... Start Fitting " + str(self.currNumNode) + " nd node, parent id : "+str(parentId)+", parent edge :"+str(parent_edge)+" ...... \n")
        start = time.time()

        ## Check terminate condition
        # dataset is empty
        if (currentData.shape[0] == 1):
            print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            print("Got empty dataset\n")
            return None

        curr_target, curr_target_count = np.unique(currentData[1:, -1].astype(np.float64), return_counts=True)
        targetCompostion = None
        for count in curr_target_count:
            if (targetCompostion != None):
                targetCompostion += "-" + str(count)
            else:
                targetCompostion = str(count)

        self.currNumNode += 1
        cur_idx = self.currNumNode

        # target is already homogen
        if (len(curr_target) == 1):
            self.tree.append([cur_idx, parentId, None, None, 0.0, 0.0, parent_edge, 0.0, curr_target[0], targetCompostion, None])
            print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            return

        # no target or attribute
        if (currentData.shape[1] < 2):
            max_count, selected_class = self.selectMajorityClass(curr_target, curr_target_count)
            self.tree.append([cur_idx, parentId, None, None, 0.0, 0.0, parent_edge, 0.0, selected_class, targetCompostion, None])
            print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            return
        ## Check terminate condition

        splitter = SplitAttribute(self.attributes_info, currentData, curr_target)
        splitter.doSplit()

        cur_attr_name = splitter.selected_attr_name
        cur_attr_idx = splitter.selected_attr_idx
        cur_threshold = splitter.selected_splitter
        cur_gain = splitter.selected_gain
        cur_split_info = splitter.selected_split_info
        thresholdCompostion = None
        if (len(cur_threshold) == 1):
            no_header_data = currentData[1:, :]
            cur_threshold = cur_threshold[0]
            # fit left child
            less_eq_data = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) <= cur_threshold.astype(np.float64))]
            less_eq_data = np.concatenate(([currentData[0, :]], less_eq_data), axis=0)
            XXY_new2 = np.delete(less_eq_data, cur_attr_idx, 1)
            thresholdCompostion = str(XXY_new2.shape[0]-1)
            self.fitInternal(XXY_new2, cur_idx, '<=' + str(cur_threshold))
            print("left child fit done. parent node: " + str(cur_attr_name)+"\n")

            # fit right child
            greater_data = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) > cur_threshold.astype(np.float64))]
            greater_data = np.concatenate(([currentData[0, :]], greater_data), axis=0)
            XXY_new2 = np.delete(greater_data, cur_attr_idx, 1)
            thresholdCompostion += "-" + str(XXY_new2.shape[0]-1)
            self.fitInternal(XXY_new2, cur_idx, '>' + str(cur_threshold))
            print("right child fit done. parent node: " + str(cur_attr_name)+"\n")

        else:
            no_header_data = currentData[1:, :]
            for i in range(len(cur_threshold)):
                selected_dataset = no_header_data[np.where(no_header_data[:, cur_attr_idx] == cur_threshold[i])]
                selected_dataset = np.concatenate(([currentData[0, :]], selected_dataset), axis=0)
                selected_dataset = np.delete(selected_dataset, cur_attr_idx, 1)
                thresholdCompostion = str(selected_dataset.shape[0]-1) if thresholdCompostion == None else thresholdCompostion+"-"+str(selected_dataset.shape[0]-1)
                self.fitInternal(selected_dataset, cur_idx, cur_threshold[i])
            print("left child fit done. parent node: " + str(cur_attr_name) + "\n")


        # save to tree. format : [treeIdx, parentId, attrName, colIndex, gain, splitInfo, parentEdge, threshold, leaveVal, targetCompostion]
        col_idx = None if cur_attr_name == None else self.attributes_info[cur_attr_name]['col_idx']
        self.tree.append(
            [cur_idx, parentId, cur_attr_name, col_idx, cur_gain,
             cur_split_info, parent_edge, cur_threshold, None, targetCompostion, thresholdCompostion])

        print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done in "+str(time.time() - start)+"!!!\n")

    def selectMajorityClass(self, curr_target, curr_target_count):
        max_count, selected_class = 0, 0
        for i in range(len(curr_target)):
            if (curr_target_count[i] > max_count):
                max_count = curr_target_count[i]
                selected_class = curr_target[i]
        return max_count, selected_class









