import math
import numpy as np


def constructFeatureDesc(XX, threshold):
    # "data_type" -> nominal=0 ; numeric=1;
    features_desc = {}

    for i in range(XX.shape[1]):
        single_desc = {'data_type': 0}
        unique, counts = np.unique(XX[1:, i].astype(np.float64), return_counts=True)
        single_desc['name'] = XX[0, i]
        if len(unique) > threshold:
            single_desc['data_type'] = 1
        single_desc['unique_edge'] = unique
        single_desc['unique_counts'] = counts
        features_desc[XX[0, i]] = single_desc
    return features_desc


def calculateEntropy(pcs):
    result = 0.0
    for pc in pcs:
        if (pc == 0 or pc == 0.0):
            continue
        else:
            result += -1 * (pc * math.log2(pc))
    return result


def calculateInfos(T):
    # T[0] = [13/14, [8,5]]
    infos = 0.0
    for i in range(len(T)):
        data = T[i]
        infos += data[0] * calculateEntropy(data[1])
    return infos


def calculateSplitInfos(T):
    # T[0] = [13/14, [8,5]]
    infos = 0.0
    for i in range(len(T)):
        data = T[i]
        infos += data[0] * calculateEntropy(data[1])
    return infos


def calculateGainRationNumericInternal(XY, current_edge, epx):
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
    current_info = calculateInfos(data_info)
    current_gain = epx - current_info

    return [current_edge, current_gain, countLessThanEq.shape[0], countGreatherThan.shape[0]]


def calculateGainRationNumeric(XY, target):
    print("calculateGainRationNumeric ")
    # calculate entropy(X)
    unique_edge = np.unique(XY[:, 0].astype(np.float64), return_counts=False)
    px = []
    for i in target:
        px.append(XY[np.where(XY[:, 1].astype(np.float64) == i)].shape[0] / XY.shape[0])
    epx = calculateEntropy(px)
    print("epx " + str(epx) + " px " + str(px))

    # gain_mat = [[selected_edge,selected_gain,selected_countLessThanEq,selected_countGreatherThan]]
    gain_mat = []
    selected_edge = 0.0
    selected_gain = 0.0
    selected_countLessThanEq = 0
    selected_countGreatherThan = 0
    for i in range(len(unique_edge)):
        current_edge = unique_edge[i]

        curr_result = calculateGainRationNumericInternal(XY, current_edge, epx)
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
            selected_countLessThanEq = countLessThanEq
            selected_countGreatherThan = countGreatherThan

    split_info = calculateEntropy([selected_countLessThanEq / XY.shape[0], selected_countGreatherThan / XY.shape[0]])
    print("calculateGainRationNumeric " + " done")
    return {'gain': selected_gain, 'selected_splitter': selected_edge, 'split_info': split_info}


def calculateGainRatio(XY, target, data_type):
    print("calculateGainRatio")
    result = {}
    if (data_type == 0):
        result = {}
    elif (data_type == 1):
        result = calculateGainRationNumeric(XY, target)
    return result


def selectNextNode(XXY, target, features_desc):
    print("select node")
    XX = XXY[:, :-1]
    Y = XXY[:, -1]

    selectedNode = {'name': None, 'selected_splitter': 0.0, 'gain_ratio': 0.0}

    raw_gain_ratios = []
    for i in range(XX.shape[1]):
        print("calculating gain ratio fitur:" + XX[0, i])
        desc = features_desc[XX[0, i]]
        XY = np.array([XX[1:, i], Y[1:]]).transpose()
        raw_gain_ratio = calculateGainRatio(XY, target, desc['data_type'], )
        raw_gain_ratios.append(raw_gain_ratio)
        print(raw_gain_ratio)

    for i in range(XX.shape[1]):
        raw_gain_ratio = raw_gain_ratios[i]
        if (raw_gain_ratio['split_info'] == 0.0):
            gain_ratio = 0.0
        else:
            gain_ratio = raw_gain_ratio['gain'] / raw_gain_ratio['split_info']
        if (gain_ratio > selectedNode['gain_ratio']):
            selectedNode = {'name': XX[0, i], 'index': i, 'selected_splitter': raw_gain_ratio['selected_splitter'],
                            'gain_ratio': gain_ratio}

    print("select node done")
    print(selectedNode)
    return selectedNode


def fitInternal(XXY, target, features_desc):
    print("")
    print("")
    print("fitting internal")
    curr_target, curr_target_count = np.unique(XXY[1:, -1].astype(np.float64), return_counts=True)

    # dataset is empty
    if (XXY.shape[0] == 1):
        return None

    # target is already homogen
    if (len(curr_target) == 1):
        return {'leave': curr_target[0]}

    # no target or attribute
    if (XXY.shape[1] < 2):
        # print(curr_target)
        # print(curr_target, curr_target_count)
        max_count = 0
        selected_class = 0
        for i in range(len(curr_target)):
            if (curr_target_count[i] > max_count):
                max_count = curr_target_count[i]
                selected_class = curr_target[i]
        return {'leave': selected_class}

    next_node = selectNextNode(XXY, target, features_desc)
    print("parent node : " + next_node['name'])
    if ('selected_splitter' in next_node):
        # XXY_new1 = np.delete(XXY, next_node['index'], 1)
        edges = {}
        XXY_cont = XXY[1:, :]

        XXY_new1 = XXY_cont[
            np.where(XXY_cont[:, next_node['index']].astype(np.float64) <= next_node['selected_splitter'])]
        XXY_new1 = np.concatenate(([XXY[0, :]], XXY_new1), axis=0)
        XXY_new2 = np.delete(XXY_new1, next_node['index'], 1)
        lessThanEqNode = fitInternal(XXY_new2, target, features_desc)
        if (lessThanEqNode != None):
            edges['<=' + str(next_node['selected_splitter'])] = lessThanEqNode
        print("first fit done. parent node: " + next_node['name'])

        XXY_new1 = XXY_cont[
            np.where(XXY_cont[1:, next_node['index']].astype(np.float64) > next_node['selected_splitter'])]
        XXY_new1 = np.concatenate(([XXY[0, :]], XXY_new1), axis=0)
        XXY_new2 = np.delete(XXY_new1, next_node['index'], 1)
        greaterThanNode = fitInternal(XXY_new2, target, features_desc)
        if (greaterThanNode != None):
            edges['>' + str(next_node['selected_splitter'])] = greaterThanNode
        print("second fit done. parent node: " + next_node['name'])

    nodes = {'node_info': next_node, 'edges': edges}
    print(nodes)
    return nodes


def fit(XXY):
    features_desc = constructFeatureDesc(XXY[:, :-1], 100)
    target, count = np.unique(XXY[1:, -1].astype(np.float64), return_counts=True)

    result = fitInternal(XXY, target, features_desc)
    print("fit done")
    return result

