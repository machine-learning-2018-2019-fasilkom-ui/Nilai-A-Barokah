import math
import time
import numpy as np
from multiprocessing import Pool

class FeatureSelector:
    def __init__(self, pool_size, attributes_info, currentData, curr_target):
        self.pool_size = pool_size

        self.attributes_info = attributes_info
        self.currentData = currentData
        self.curr_target = curr_target

        self.selected_attr_name = None
        self.selected_attr_idx = None
        self.selected_splitter = None
        self.selected_gain = 0.0
        self.selected_split_info = 0.0
        self.selected_gain_ratio = 0.0

    def doSelect(self):
        numOfCurrAttr = self.currentData[:, :-1].shape[1]

        listParam = []
        for i in range(numOfCurrAttr):
            currAttrName = self.currentData[0, i]
            start = time.time()
            #print("Calculating gain ratio fitur: " + currAttrName + " ....")

            desc = self.attributes_info[currAttrName]
            XY = np.array([self.currentData[1:, i], self.currentData[1:, -1]]).transpose()
            listParam.append((XY, desc['data_type'], self.currentData[0, i], i))

        with Pool(self.pool_size) as p:
            gainAll = p.starmap(self.gainRatio, listParam)
            p.close()
            p.join()

        for i in range(len(gainAll)):
            if (gainAll[i] != None):
                gain = gainAll[i][0]
                split_info = gainAll[i][1]
                edge_splitter = gainAll[i][2]
                attr_name = gainAll[i][3]
                attr_idx = gainAll[i][4]

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
        #print("")

    def gainRatio(self, XY, data_type, attr_name, attr_idx):
        if (data_type == 0):
            gain, split_info, edge_splitter = self.gainRatioNominal(XY)
        elif (data_type == 1):
            gain, split_info, edge_splitter = self.gainRatioNumeric(XY)
        return [gain, split_info, edge_splitter, attr_name, attr_idx]

    def gainRatioNominal(self, XY):
        unique_edge = np.unique(XY[:, 0], return_counts=False)
        px = []
        for i in self.curr_target:
            px.append(XY[np.where(XY[:, 1].astype(np.float64) == i)].shape[0] / XY.shape[0])
        epx = self.entropy(px)

        gain = 0.0
        split_info_conprob = []
        for i in range(len(unique_edge)):
            split = XY[np.where(XY[:, 0] == unique_edge[i])]
            sub_conprob = []
            for j in range(len(self.curr_target)):
                sub_split = split[np.where( split[:,-1].astype(np.float64) == self.curr_target[j])]
                sub_conprob.append(sub_split.shape[0]/split.shape[0])
            gain += self.entropy(sub_conprob)*split.shape[0]/XY.shape[0]
            split_info_conprob.append(split.shape[0]/XY.shape[0])

        split_info = self.entropy(split_info_conprob)
        gain = epx - gain
        return gain, split_info, unique_edge

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
        for i in range(len(unique_edge)):
            current_edge = unique_edge[i]

            curr_result = self.gainRatioNumericInternal(XY, current_edge, epx)
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
        return selected_gain, selected_split_info, [selected_edge]

    def gainRatioNumericInternal(self, XY, current_edge, epx):
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