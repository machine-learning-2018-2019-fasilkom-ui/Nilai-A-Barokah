class Node:
    def __init__(self, name, childs, parent_edge, edges_string, class_composition_string, edge_composition_string, feature_type, leave_value):
        self.name = name
        self.childs = childs # None if leaf
        self.parent_edge = parent_edge # None if root
        self.edges = str(edges_string).split("|")

        self.class_composition = str(class_composition_string).split("|")
        self.edge_composition = str(edge_composition_string).split("|")
        self.feature_type = feature_type
        self.leave_value = leave_value

    def isRootNode(self):
        return self.parent_edge == None

    def isLeaveNode(self):
        return self.childs == None