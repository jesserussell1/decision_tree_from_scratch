# tree_builder.py

# Load packages
import numpy as np
from node import Node

# Create a class to build the decision tree
class TreeBuilder:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    # Recursive function to build the decision tree
    def build_tree(self, X, y, depth=0):
        # Base cases: if all labels are the same or max depth is reached
        if len(np.unique(y)) == 1:
            return Node(value=np.unique(y)[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._most_common_class(y))

        # Find the best split
        best_feature, best_threshold, best_score = None, None, float('inf')
        best_left_X, best_right_X, best_left_y, best_right_y = None, None, None, None

        # Loop over each feature and each threshold to find the best split
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                score = self._calculate_gini(left_y, right_y)

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_left_X, best_right_X = X[left_mask], X[right_mask]
                    best_left_y, best_right_y = left_y, right_y

        # Recursively build left and right subtrees
        left_node = self.build_tree(best_left_X, best_left_y, depth + 1)
        right_node = self.build_tree(best_right_X, best_right_y, depth + 1)

        # Return the best split
        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    # Function to calculate the Gini impurity
    def _calculate_gini(self, left_y, right_y):
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        # Method to calculate the Gini impurity
        def gini(y):
            unique_classes, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

        left_gini = gini(left_y)
        right_gini = gini(right_y)

        # Return the weighted average of the Gini impurities
        return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini

    # Function to find the most common class of a branch
    def _most_common_class(self, y):
        return np.bincount(y).argmax()
