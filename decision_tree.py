# decision_tree.py

# Load packages
import numpy as np
from tree_builder import TreeBuilder

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.tree_builder = TreeBuilder(max_depth=max_depth)
        self.root = None

    def fit(self, X, y):
        self.root = self.tree_builder.build_tree(X, y)
        return self  # Returning self for method chaining

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        node = self.root
        while node.value is None:  # Traverse until we reach a leaf node
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def print_tree(self, node=None, depth=0):
        """Recursively print the decision tree."""
        if node is None:
            node = self.root  # If no node is passed, start from the root

        indent = "|   " * depth
        if node.value is not None:  # Leaf node
            print(f"{indent}Leaf: Class={node.value}")
        else:  # Decision node
            print(f"{indent}Feature {node.feature} <= {node.threshold:.4f}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
