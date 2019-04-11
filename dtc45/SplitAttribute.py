import math
import time
import numpy as np

class SplitAttribute:
    def __init__(self, attributes_info, currentData, curr_target):
        self.attributes_info = attributes_info
        self.currentData = currentData
        self.curr_target = curr_target

        self.selected_attr_name = None
        self.selected_attr_idx = -1
        self.selected_splitter = -1.0
        self.selected_gain = -1.0
        self.selected_split_info = -1.0
        self.selected_gain_ratio = -1.0

    def doSplit(self):
        numOfCurrAttr = self.currentData[:, :-1].shape[1]

        for i in range(numOfCurrAttr):
            currAttrName = self.currentData[0, i]
            start = time.time()
            print("Calculating gain ratio fitur: " + currAttrName + " ....")

            desc = self.attributes_info[currAttrName]
            XY = self.currentData[1:, :]
            self.gainRatio(XY, desc['data_type'], self.currentData[0, i], i)
            #print("Done calculating gain ratio fitur:" + currAttrName + " in " + str(start - time.time()))

    def gainRatio(self, XY, data_type, attr_name, attr_idx):
        if (data_type == 0):
            # TODO implementation
            return
        elif (data_type == 1):
            gain, split_info, edge_splitter = self.gainRatioNumeric(XY)
            if (split_info == 0.0):
                gain_ratio = 0.0
            else:
                gain_ratio = gain / split_info

            if (gain_ratio > self.selected_gain_ratio):
                self.selected_attr_name = attr_name
                self.selected_attr_idx = attr_idx
                self.selected_splitter = edge_splitter
                self.selected_gain = gain
                self.selected_split_info = split_info
                self.selected_gain_ratio = gain_ratio

    def gainRatioNumeric(self, XY):
        #print("calculateGainRationNumeric ")
        unique_edge = np.unique(XY[:, 0].astype(np.float64), return_counts=False)
        px = []
        for i in self.curr_target:
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