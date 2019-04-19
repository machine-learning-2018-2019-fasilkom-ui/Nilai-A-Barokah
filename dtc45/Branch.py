class Branch:
    def __init__(self, branchType, nominalValue, continueValue, label, isLeafNode):
        self.branchType = branchType
        self.nominalValue = nominalValue
        self.continueValue = continueValue
        self.label = label
        self.isLeafNode = isLeafNode