class Node:
    def __init__(self, child, parent, isLeafNode, branches, weight, errorEstimate, indexOfDataset):
        self.child = child
        self.parent = parent
        self.isLeafNode = isLeafNode
        self.branches = branches
        self.weight = weight
        self.errorEstimate = errorEstimate