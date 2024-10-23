from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


from main import load_data


def create_decision_tree(X_train, Y_train, max_depth=None, criterion='gini'):
    """
    Create and train a decision tree classifier using scikit-learn.
    
    :param X_train: Training features (input data), a 2D array or DataFrame.
    :param Y_train: Training labels (output data), a 1D array or Series.
    :param max_depth: (Optional) Maximum depth of the tree. If None, the tree grows until leaves are pure or 
                      until all leaves contain less than min_samples_split samples.
    :param criterion: (Optional) The function to measure the quality of a split. Supported criteria are 
                      "gini" for the Gini impurity and "entropy" for the information gain.
    
    :return: The trained decision tree model.
    """
    # Initialize the decision tree classifier
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    
    # Fit the decision tree model on the training data
    tree_clf.fit(X_train, Y_train)
    
    # Print tree depth and number of leaf nodes
    print(f"Tree Depth: {tree_clf.tree_.max_depth}")
    print(f"Number of Leaf Nodes: {tree_clf.tree_.n_leaves}")

    # Visualize the tree
    plt.figure(figsize=(20, 10))  # You can adjust the figure size as needed
    plot_tree(tree_clf, filled=True, rounded=True, feature_names=None, class_names=None)
    plt.show()

    return tree_clf

X, y = load_data('wifi_db/noisy_dataset.txt')
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Create and train the decision tree
model = create_decision_tree(X_train, Y_train, max_depth=3, criterion='gini')

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")
