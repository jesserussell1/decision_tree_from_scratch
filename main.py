# main.py

# Load packages
from decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd

# First try it out the relatively simple Iris dataset
print("Create and test a tree on the Iris dataset.\n")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and train the classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Calculate the training set accuracy
accuracy = clf.accuracy(X_train, y_train)
print(f"\nTraining Accuracy: {accuracy:.4f}")

# Calculate test set accuracy
accuracy = clf.accuracy(X_test, y_test)
print(f"Accuracy on Test Data: {accuracy:.4f} \n")

# Print the tree
#print("Decision Tree Structure:")
#clf.print_tree()


# Now try it with the much more complex adult income dataset
print("Create and test a tree on the adult income dataset.\n")

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Convert the data to Pandas
data = pd.read_csv(url, names=columns, sep=r',\s*', engine='python')

# Handle Missing Values
data = data.dropna()

# One-hot encode categorical columns
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the Data into Features and Labels
X = data_encoded.drop('income_>50K', axis=1)  # Features (excluding the target variable)
y = data_encoded['income_>50K']  # Target variable (binary classification)

# Convert X and y to NumPy arrays
X = X.values  # Convert the DataFrame to a NumPy array
y = y.values  # Convert the target column to a NumPy array

# Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Calculate the training set accuracy
accuracy = clf.accuracy(X_train, y_train)
print(f"\nTraining Accuracy: {accuracy:.4f}")

# Calculate test set accuracy
accuracy = clf.accuracy(X_test, y_test)
print(f"Accuracy on Test Data: {accuracy:.4f} \n")

# Print the decision tree structure
# clf.print_tree()