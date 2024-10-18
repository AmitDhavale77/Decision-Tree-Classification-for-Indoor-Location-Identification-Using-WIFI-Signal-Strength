import matplotlib.pyplot as plt
import numpy as np
from binary_node import Binarytree

def load_data(filename):
    """ Load data from input file

    Return X, Y arrays
    @aps124
    """

    pass


def normalise_data(data):
    """  Normalise the data
    @???
    """

def create_node(attribute, value, left, right, leaf=False):
    """
    @amit - convert to use class for node
    """

    binary_node = Binarytree(attribute, value, left, right, leaf=False)

    return binary_node


def calculate_entropy(data):
    """
    data : np.array with last column indicating classes
    @ola
    """
    if data.ndim == 1:
        return 0 # if there is only one instance then entropy is 0 
    else:
        dict_classes = {} # dictionary counting how many elements are in particular classes
        total = len(data) # store total number of instances
        for row in data:
            class_row = row[-1] # store class of particular instance (row)
            if class_row in dict_classes:
                dict_classes[class_row] += 1
            else:
                dict_classes[class_row] = 1
        
        entropy = 0
        for element in dict_classes.values():
            entropy -= ((element/total)*np.log2(element/total)) # calculate entropy

        return entropy


def calculate_information_gain(dataset, subsets):
    """
    @ola - related to above
    see if integral option is better (we think no but worth checking?)
    
    dataset : np.array with last column indicating classes

    subsets : array of k subsets of dataset
    """

    ig = calculate_entropy(dataset) 
    total_length = len(dataset)
    for subset in subsets:
        ig -= (len(subset)/total_length) * calculate_entropy(subset)

    return ig


def find_split(dataset):
    """
    This function is the big one. It should determine the optimal attribute and value to split on
    @lauren to investigate and possibly delegate

    Args: dataset - TODO decide on data structure

    Returns: (feature, (split_point, information_gain))
                where split point is the value in the feature to split on 
                so left split is values <= split_point and 
                right split is values > split_point
    """
    # calculate information gain at each potential split point for each feature
    best_feature_split = []
    for feature in dataset:
        feature_split_ig = []
        for split_value in feature:
            # split data into two groups based on split point
            pass
            # calculate information gain for each split point (feature data < value)
            split_point_ig = calculate_information_gain(split_data)
            feature_split_ig.append((split_value, split_point_ig))
        # select feature with highest information gain
        best_split_for_feature = max(feature_split_ig, key=lambda x: x[1])
        # add best split for feature to list
        best_feature_split.append((feature, best_split_for_feature))
    # select feature with highest information gain
    optimum_split = max(best_feature_split, key=lambda x: x[1][1])
    
    return optimum_split



def split_data(data, split_value):
    """ This should take in results from above and acutally split the data
    @lauren

    Args: data - TODO decide on data structure
        split_value - value to split on (left split is values <= split_point and right split is values > split_point)

    Returns: left_data, right_data
    """

    pass


def decision_tree_learning(X, Y, depth=0, max_depth=None):
    """ This code is a placeholder
    Once we do the other parts, can tackle this bit

    Args:
        X train
        Y train,
        depth, current depth
        max depth = how far should we go?

    Returns:
        tree (nested dictionary representing entire tree and branches and leafs)
    """

    if depth == max_depth:
        return

    # If all data in dataset has the same label, create leaf node
    if all(Y) is the same:
        return create_node(None, None, None, None, leaf_val=whatever Y is)
    
    split_attribute, split_value = find_split(X, Y)
    Xleft, Yleft, Xright, Yright = split_data(X, Y, split_attribute, split_value)
    
    left_branch, left_depth = decision_tree_learning(Xleft, Yleft, depth + 1)
    right_branch, right_depth = decision_tree_learning(Xright, Yright, depth + 1)

    node = create_node(split_attribute, split_value, left_branch, right_branch)

    return node, max(left_depth, right_depth)


def visualize_tree():
    """ BONUS FUNCTION: Plot tree visualization

    Tackle this later
    """

    pass


def evaluate_tree(test_data, trained_tree):
    """ Evaluate accuracy of trained tree using test data
    @aanish
    """

    pass


def prune_tree():
    """ wait till next week's lecture to see how to implement this
    """

    pass


if __name__ == "__main__":
    pass
