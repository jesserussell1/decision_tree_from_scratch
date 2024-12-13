import numpy as np

# Create a TreeNode class
# It contains information about the feature to split on,
# the threshold value, and the left and right subtrees
# This class is used by the DecisionTreeClassifier class
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature to split on
        self.threshold = threshold  # Threshold value to split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Leaf value (class label)

# Create a DecisionTreeClassifier class
# This class contains methods to train and predict using a decision tree
# It recursively builds the tree and makes predictions, using the TreeNode class
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the tree, set when creating the tree
        self.root = None # Root of the tree, set when training the tree

    # Train the decision tree
    # Calls the recursive _build_tree method
    # Sets the root of the tree to the returned TreeNode
    def fit(self, X, y):
        """Train the decision tree."""
        self.root = self._build_tree(X, y)

    # Recursively build the decision tree
    # Calls the recursive _build_tree method and stops when a leaf node is reached
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Stop if only one class or max depth reached
        # or if all samples belong to the same class
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return TreeNode(value=unique_classes[0])

        # Best split initialization
        # These are set to None to be initialized later
        best_feature, best_threshold = None, None
        best_left_y, best_right_y = None, None
        best_left_X, best_right_X = None, None
        best_score = float('inf')

        # Iterate over each feature and find the best split
        # The best split is the one with the lowest Gini impurity
        # The Gini impurity is calculated by the _calculate_gini method
        # Loop over each feature and threshold
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_X, right_X = X[left_mask], X[right_mask]
                left_y, right_y = y[left_mask], y[right_mask]

                # Calculate the Gini impurity or another metric for the split
                score = self._calculate_gini(left_y, right_y)

                # Update the best split if the current split has a lower Gini impurity
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_left_X, best_right_X = left_X, right_X
                    best_left_y, best_right_y = left_y, right_y

        # Create left and right subtrees recursively
        # Calls the recursive _build_tree method
        # Sets the left and right subtrees to the returned TreeNode
        left_node = self._build_tree(best_left_X, best_left_y, depth + 1)
        right_node = self._build_tree(best_right_X, best_right_y, depth + 1)

        # Return the TreeNode with the best split
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    # Calculate the Gini impurity for a split
    def _calculate_gini(self, left_y, right_y):
        """Calculate the Gini impurity for a split."""
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        # Calculate the Gini impurity
        # This represents the quality of the split, how homogeneous it is
        def gini(y):
            unique_classes, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

        left_gini = gini(left_y)
        right_gini = gini(right_y)

        return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini

    # For a tree, predict the class labels for new samples
    def predict(self, X):
        """Predict the class labels for samples in X."""
        predictions = [self._predict_sample(sample) for sample in X]
        return np.array(predictions)

    # For a tree, predict the class label for a single sample
    def _predict_sample(self, sample):
        """Predict a single sample."""
        node = self.root
        while node.value is None:  # Traverse until a leaf node is reached
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    # Print the tree structure recursively
    def print_tree(self, node=None, depth=0):
        """Recursively print the tree structure."""
        if node is None:
            node = self.root  # Start from the root

        if node.value is not None:  # Leaf node
            print(f"{'|   ' * depth}Leaf: Class={node.value}")
        else:  # Decision node
            print(f"{'|   ' * depth}Feature {node.feature} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    # For a tree, calculate the accuracy
    def accuracy(self, X, y):
        """Calculate the accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Create a toy dataset
# Set random seed for reproducibility
np.random.seed(42)

# Generate a dataset with 100 samples and 4 features
X = np.random.rand(100, 4)  # Random features between 0 and 1

# Generate binary labels (0 or 1) for classification
y = np.random.choice([0, 1], size=100)  # Random binary labels


# Instantiate the DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3)

# Fit the model to the larger dataset
tree.fit(X, y)

# Print the tree structure
print("Decision Tree Structure:")
tree.print_tree()

# Calculate accuracy on the training dataset
accuracy = tree.accuracy(X, y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Predict new samples
new_samples = np.array([
    [0.5, 0.3, 0.8, 0.2],
    [0.7, 0.6, 0.1, 0.4]
])
predictions = tree.predict(new_samples)
print("\nPredictions for new samples:", predictions)