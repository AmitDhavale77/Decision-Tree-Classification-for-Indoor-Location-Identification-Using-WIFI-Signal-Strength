import numpy as np
from data_prep import sort_data, split_data


def create_node(feature, value, left, right):
    """Given node attributes, instantiate the node as a dictionary"""
    node = {
        "feature": feature,
        "value": value,
        "left": left,
        "right": right,
    }
    return node


def calculate_entropy(data):
    """Calculate the entropy of a given array

    data : np.array with 1 dimension
    """
    if len(data) == 1:
        return 0  # if there is only one instance then entropy is 0
    else:
        dict_classes = (
            {}
        )  # dictionary counting how many elements are in particular classes
        total = len(data)  # store total number of instances
        for class_row in data:
            if class_row in dict_classes.keys():
                dict_classes[class_row] += 1
            else:
                dict_classes[class_row] = 1

        entropy = 0
        for element in dict_classes.values():
            entropy -= (element / total) * np.log2(element / total)  # calculate entropy

        return entropy


def calculate_information_gain(Y, subsets):
    """Calculate the information gain of comparing Y with subsets (Yleft, Yright)

    dataset : np.array with last column indicating classes

    subsets : array of k subsets of dataset
        subsets = (Yleft, Yright)
    """
    ig = calculate_entropy(Y)

    total_length = len(Y)
    for subset in subsets:
        ig -= (len(subset) / total_length) * calculate_entropy(subset)

    return ig


def find_split(X, Y):
    """Given input dataset X and Y, determine the optimal way to split the data
    by iteratively calculating information gain and finding the maximum

    Args: X, Y

    Returns: best_feature, best_value
    """
    # calculate information gain at each potential split point for each feature
    max_info_gain = 0
    best_feature = None
    best_value = None

    # iterate through each feature
    for feature in range(X.shape[1]):

        # Sort X by feature
        X_sorted, Y_sorted = sort_data(X, Y, feature)

        # Get midpoints of the feature column of sorted X
        midpoints = get_midpoints(X_sorted, feature)
        for midpoint in midpoints:
            # Split data into two groups based on split point
            _, Y_left, _, Y_right = split_data(X_sorted, Y_sorted, feature, midpoint)

            # Calculate information gain for each split point (feature data < value)
            split_point_ig = calculate_information_gain(Y_sorted, (Y_left, Y_right))

            # If new split point is larger than the previous maximum, set it to be
            # the current best split option
            if split_point_ig > max_info_gain:
                max_info_gain = split_point_ig
                best_feature = feature
                best_value = midpoint

    return best_feature, best_value


def get_midpoints(X, feature):
    """
    Calculate the midpoints between consecutive values in the sorted feature data
    Args:
        X (array) : sorted array of data
        feature (int) - the feature to calculate midpouints for

    Returns:
        array of midpoints of length len(X[:, feature]) - 1
    """
    # filter for feature array only
    X_feature = X[:, feature]
    # calculate midpoints between consecutive values
    midpoints = [
        (X_feature[i] + X_feature[i + 1]) / 2 for i in range(len(X_feature) - 1)
    ]

    midpoints = np.array(list(set(midpoints)))  # Remove duplicates
    return midpoints


def check_all_elements_same(arr):
    """Check if all elements in a numpy array are the same"""

    if np.all(arr == arr[0]):
        return True, arr[0]
    else:
        return False, None


def decision_tree_learning(X, Y, depth=0, max_depth=None):
    """Train the decision tree on dataset X and Y

    Args:
        X train
        Y train,
        depth, current depth
        max depth = how far should we go?

    Returns:
        tree (nested dictionary representing entire tree and branches and leafs)
        depth (int representing how many levels the tree has)
    """

    # End decision tree learning if max depth has been met
    if depth == max_depth:
        return

    # Check if all Y has the same label and if so create a leaf node
    y_has_same_label, label_value = check_all_elements_same(Y)
    if y_has_same_label:
        leaf_node = create_node(None, label_value, None, None)
        return leaf_node, depth  # Set leaf value to the label of Y

    # Find optimal split feature by searching for optimal information gain
    split_feature, split_value = find_split(X, Y)
    # print(f"Depth: {depth}: Splitting on X[{split_feature}] > {split_value}")
    # Split data on optimal feature into left and right datasets
    X_left, Y_left, X_right, Y_right = split_data(X, Y, split_feature, split_value)

    # Make recursive calls to train sub-trees on left and right datasets
    left_branch, left_depth = decision_tree_learning(
        X_left, Y_left, depth + 1, max_depth=max_depth
    )
    right_branch, right_depth = decision_tree_learning(
        X_right, Y_right, depth + 1, max_depth=max_depth
    )

    # Create node with current split feature and value
    node = create_node(split_feature, split_value, left_branch, right_branch)

    return node, max(left_depth, right_depth)


def count_leaves(tree):
    """Calculate number of leaves i.e max span of the tree"""

    if tree is None:
        return 0
    elif tree["feature"] is None:
        return 1

    return count_leaves(tree["left"]) + count_leaves(tree["right"])


def get_tree_depth(tree):
    """Calculate tree depth"""
    if tree is None or tree["feature"] is None:
        return 0
    return 1 + max(get_tree_depth(tree["left"]), get_tree_depth(tree["right"]))


def predictions(tree, test_x):
    """
    Predict output values from a tree using a test set

    Args:
        tree(dict): trained decision tree dictionary
        test_x(np.array): array of x features to get classification predictions from the tree

    Returns:
        np.array: array of size len(test_x) of output predictions from tree

    """

    num_rows, _ = test_x.shape
    predictions = np.zeros((num_rows,))

    for row in range(num_rows):
        test_row = test_x[row, :]
        predictions[row] = row_predict(tree, test_row)

    return predictions


def row_predict(tree, test_row):
    """
    Make a prediction for a single row on an input dataset using a trained tree

    Args:
        tree(dict): trained decision tree dictionary
        test_row(np.array): single row of an np.array to make a decision from
    """

    if tree["feature"] is None:
        return tree["value"]

    cur_feature = tree["feature"]
    cur_value = tree["value"]
    go_right = test_row[cur_feature] > cur_value
    if go_right:
        return row_predict(tree['right'], test_row)
    else:
        return row_predict(tree['left'], test_row)