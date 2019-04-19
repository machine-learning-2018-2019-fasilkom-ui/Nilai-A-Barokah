import math
import time
import numpy as np


class C45v2back:
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
            unique, counts = np.unique(self.dataset[1:, i].astype(np.float64), return_counts=True)
            single_desc['name'] = self.dataset[0, i]
            if len(unique) > threshold:
                single_desc['data_type'] = 1
            single_desc['unique_edge'] = unique
            single_desc['unique_counts'] = counts
            single_desc['col_idx'] = i
            self.attributes_info[self.dataset[0, i]] = single_desc

    def fit(self):
        start = time.time()
        print("...... Start fitting "+str(time.ctime(int(start)))+"...... \n")

        self.fitInternal(self.dataset, 0, None, None)

        print("\n\n\n")
        print("current time :" + str(time.ctime(int(start))) + "\n")
        print("current time :"+str(time.ctime(int(time.time())))+"\n")
        print("Fitting done in  " + str(time.time() - start))

    def fitInternal(self, currentData, parentId, parent_edge, parent_edge_operator):
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
                targetCompostion += "|" + str(count)
            else:
                targetCompostion = str(count)

        self.currNumNode += 1
        cur_idx = self.currNumNode

        # target is already homogen
        if (len(curr_target) == 1):
            self.tree.append([cur_idx, parentId, None, None, 0.0, 0.0, parent_edge, parent_edge_operator, 0.0, curr_target[0], targetCompostion, None])
            print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            return

        # no target or attribute
        if (currentData.shape[1] < 2):
            max_count, selected_class = self.selectMajorityClass(curr_target, curr_target_count)
            self.tree.append([cur_idx, parentId, None, None, 0.0, 0.0, parent_edge, parent_edge_operator, 0.0, selected_class, targetCompostion, None])
            print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done!!!\n")
            return
        ## Check terminate condition

        cur_attr_name, cur_attr_idx, cur_edge, cur_threshold, cur_gain, cur_split_info = self.splitAttribute(currentData, curr_target)
        cur_edge_comparator = None


        if (cur_threshold != None):
            no_header_data = currentData[1:, :]

            # fit left child
            less_eq_data = no_header_data[np.where(no_header_data[:, cur_attr_idx].astype(np.float64) <= cur_threshold)]
            less_eq_data = np.concatenate(([currentData[0, :]], less_eq_data), axis=0)
            XXY_new2 = np.delete(less_eq_data, cur_attr_idx, 1)
            thresholdCompostion = str(XXY_new2.shape[0])
            self.fitInternal(XXY_new2, cur_idx, cur_threshold, '<=')
            print("left child fit done. parent node: " + cur_attr_name+"\n")



            # fit right child
            greater_data = no_header_data[np.where(no_header_data[1:, cur_attr_idx].astype(np.float64) > cur_threshold)]
            greater_data = np.concatenate(([currentData[0, :]], greater_data), axis=0)
            XXY_new2 = np.delete(greater_data, cur_attr_idx, 1)
            thresholdCompostion += "-" + str(XXY_new2.shape[0])
            self.fitInternal(XXY_new2, cur_idx, cur_threshold, '>')
            print("right child fit done. parent node: " + cur_attr_name+"\n")

            col_idx = None if cur_attr_name == None else self.attributes_info[cur_attr_name]['col_idx']
            # save to tree. format : [treeIdx, parentId, attrName, colIndex, gain, splitInfo, parentEdge, threshold, leaveVal, targetCompostion]
            self.tree.append([cur_idx, parentId, cur_attr_name, col_idx, cur_gain,cur_split_info, parent_edge, parent_edge_operator, cur_threshold, None, targetCompostion, thresholdCompostion])

        print("......  Fitting " + str(self.currNumNode) + " nd node, parent id : " + str(parentId) + ", parent edge :" + str(parent_edge) + " ...... Done in "+str(time.time() - start)+"!!!\n")

    def splitAttribute(self, currentData, curr_target):
        selected_attr_name = None
        selected_attr_idx = -1
        selected_splitter = -1.0
        selected_gain = -1.0
        selected_split_info = -1.0
        selected_gain_ratio = -1.0

        numOfCurrAttr = currentData[:, :-1].shape[1]
        raw_gain_ratios = []
        print("number of current attribute : " + str(numOfCurrAttr))

        for i in range(0, numOfCurrAttr):
            currAttrName = currentData[0, i]
            #print( "calculating gain ratio fitur:" + currAttrName + " " + str(time.ctime(int(time.time()))) )

            desc = self.attributes_info[currAttrName]
            #XY = currentData[1:, :]
            XY = np.array([currentData[1:, i], currentData[1:,-1]]).transpose()
            gain, split_info, edge_splitter = self.gainRatio(XY, curr_target, desc['data_type'], )
            raw_gain_ratios.append([gain, split_info, edge_splitter])
            #print([gain, split_info, edge_splitter])
            #print("Done calculating gain ratio fitur:" + currAttrName+" in "+str(time.ctime(int(time.time()))))

        for i in range(numOfCurrAttr):
            gain = raw_gain_ratios[i][0]
            split_info = raw_gain_ratios[i][1]
            edge_splitter = raw_gain_ratios[i][2]

            if (split_info == 0.0):
                gain_ratio = 0.0
            else:
                gain_ratio = gain / split_info

            if (gain_ratio > selected_gain_ratio):
                selected_attr_name = currentData[0, i]
                selected_attr_idx = i
                selected_splitter = edge_splitter
                selected_gain = gain
                selected_split_info = split_info
                selected_gain_ratio = gain_ratio

        return selected_attr_name, selected_attr_idx, None, selected_splitter, selected_gain, selected_split_info

    def gainRatio(self, XY, target, data_type):
        gain, split_info, edge_splitter = 0.0, 0.0, 0.0
        if (data_type == 0):
            # TODO implementation
            gain, splitInfo, edge_splitter = 0.0, 0.0, 0.0
        elif (data_type == 1):
            gain, split_info, edge_splitter = self.gainRatioNumeric(XY, target)
        return gain, split_info, edge_splitter

    def gainRatioNumeric(self, XY, target):
        #print("calculateGainRationNumeric ")
        # calculate entropy(X)
        unique_edge = np.unique(XY[:, 0].astype(np.float64), return_counts=False)
        px = []
        for i in target:
            px.append(XY[np.where(XY[:, 1].astype(np.float64) == i)].shape[0] / XY.shape[0])
        epx = self.entropy(px)
        #print("epx " + str(epx) + " px " + str(px))

        # gain_mat = [[selected_edge,selected_gain,selected_countLessThanEq,selected_countGreatherThan]]
        gain_mat = []
        selected_edge = 0.0
        selected_gain = 0.0
        selected_count_less_eq = 0
        selected_count_greater = 0
        for i in range(len(unique_edge)-1):
            current_edge = unique_edge[i]

            curr_result = self.calculateGainRationNumeric(XY, current_edge, epx)
            gain_mat.append(curr_result)

        for i in range(len(gain_mat)):
            current_gain_mat = gain_mat[i]
            current_edge = current_gain_mat[0]
            current_gain = current_gain_mat[1]
            countLessThanEq = current_gain_mat[2]
            countGreatherThan = current_gain_mat[3]
            if (current_gain > selected_gain):
                selected_edge = current_edge
                selected_gain = current_gain
                selected_count_less_eq = countLessThanEq
                selected_count_greater = countGreatherThan

        selected_split_info = self.entropy([selected_count_less_eq / XY.shape[0], selected_count_greater / XY.shape[0]])
        # return {'gain': selected_gain, 'selected_splitter': selected_edge, 'split_info': split_info}
        return selected_gain, selected_split_info, selected_edge

    def calculateGainRationNumeric(self, XY, current_edge, epx):
        # print("calculateGainRationNumericInternal ")

        countLessThanEq = XY[np.where(XY[:, 0].astype(np.float64) <= current_edge)]
        countLessThanEqYes = countLessThanEq[np.where(countLessThanEq[:, 1].astype(np.float64) == 1)].shape[0]
        countLessThanEqNo = countLessThanEq[np.where(countLessThanEq[:, 1].astype(np.float64) == 0)].shape[0]
        countGreatherThan = XY[np.where(XY[:, 0].astype(np.float64) > current_edge)]
        countGreatherThanYes = countGreatherThan[np.where(countGreatherThan[:, 1].astype(np.float64) == 1)].shape[0]
        countGreatherThanNo = countGreatherThan[np.where(countGreatherThan[:, 1].astype(np.float64) == 0)].shape[0]

        data_info = []
        if (len(countLessThanEq) > 0):
            data_info.append([len(countLessThanEq) / XY.shape[0],
                              [countLessThanEqYes / len(countLessThanEq), countLessThanEqNo / len(countLessThanEq)]])
        if (len(countGreatherThan) > 0):
            data_info.append([len(countGreatherThan) / XY.shape[0], [countGreatherThanYes / len(countGreatherThan),
                                                                     countGreatherThanNo / len(countGreatherThan)]])
        current_info = self.infos(data_info)
        current_gain = epx - current_info

        return [current_edge, current_gain, countLessThanEq.shape[0], countGreatherThan.shape[0]]

    def entropy(self, pcs):
        result = 0.0
        for pc in pcs:
            if (pc == 0 or pc == 0.0):
                continue
            else:
                result += -1 * (pc * math.log2(pc))
        return result

    def infos(self, T):
        # T[0] = [13/14, [8,5]]
        infos = 0.0
        for i in range(len(T)):
            data = T[i]
            infos += data[0] * self.entropy(data[1])
        return infos

    def selectMajorityClass(self, curr_target, curr_target_count):
        max_count, selected_class = 0, 0
        for i in range(len(curr_target)):
            if (curr_target_count[i] > max_count):
                max_count = curr_target_count[i]
                selected_class = curr_target[i]
        return max_count, selected_class

