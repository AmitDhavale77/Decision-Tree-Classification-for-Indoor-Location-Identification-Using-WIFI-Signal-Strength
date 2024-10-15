import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    """ Load data from input file

    Return X, Y arrays
    """

    pass


def create_node(attribute, value, left, right, leaf=False):
    """
    """

    node = {
        'attribute': attribute,  # str
        'value': value,  # float
        'left': left,  # <node>
        'right': right,  # <node>
        'leaf': leaf  # False,
    }

    return node


def calculate_entropy():
    """
    """

    pass


def calculate_information_gain():
    """
    """

    pass


def find_split():
    """
    """

    pass


def split_data():
    """
    """

    pass


def decision_tree_learning(dataset, depth=0):
    """
    """

    # If all data in dataset has the same label, create leaf node
    if ...
        return create_node(None, None, None, None, leaf=True)
    
    split_attribute, split_value = find_split()
    left_data, right_data = split_data(split_attribute, split_value)
    
    node = create_node(...)

    left_branch, left_depth = decision_tree_learning(left_data, depth + 1)
    right_branch, right_depth = decision_tree_learning(right_data, depth + 1)

    return node, max(left_depth, right_depth)


def visualize_tree():
    """ BONUS FUNCTION: Plot tree visualization
    """

    pass


def evaluate_tree(test_data, trained_tree):
    """ Evaluate accuracy of trained tree using test data
    """

    pass


def prune_tree():
    """
    """

    pass


if __name__ == "__main__":
    pass
