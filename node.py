# node.py

# Class representing a node in the decision tree
# Used by the TreeBuilder class
class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature  # The feature to split on
        self.threshold = threshold  # The threshold for the split
        self.value = value  # The class label (for leaf nodes)
        self.left = left  # Left child (subtree)
        self.right = right  # Right child (subtree)
