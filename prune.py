import copy  # part of python standard library
from evaluate import simple_compute_accuracy


def prune_tree(tree, X_test, Y_test):
    """ Prunes the tree by iteratively replacing nodes with child nodes to maximize test accuracy """
    nodes_with_two_leaves = find_nodes_with_two_leaves(tree, [])
    nodes_visited = set()
    baseline_accuracy = simple_compute_accuracy(tree, X_test, Y_test)

    while nodes_with_two_leaves:
        node = nodes_with_two_leaves.pop()

        if node in nodes_visited:
            continue
        nodes_visited.add(node)

        # Evaluate both left and right prunings
        tmp_tree_left = copy.deepcopy(tree)
        tmp_tree_right = copy.deepcopy(tree)
        replace_node(tmp_tree_left, node, use_right=False)
        replace_node(tmp_tree_right, node, use_right=True)

        # Calculate accuracies for pruned versions
        left_accuracy = simple_compute_accuracy(tmp_tree_left, X_test, Y_test)
        right_accuracy = simple_compute_accuracy(tmp_tree_right, X_test, Y_test)

        # Update tree if accuracy improves
        if left_accuracy >= baseline_accuracy or right_accuracy >= baseline_accuracy:
            if left_accuracy >= right_accuracy:
                tree = tmp_tree_left
                baseline_accuracy = left_accuracy
            else:
                tree = tmp_tree_right
                baseline_accuracy = right_accuracy

            # Update prunable nodes list
            nodes_with_two_leaves = find_nodes_with_two_leaves(tree, [])
            nodes_visited = set()

    return tree

def find_nodes_with_two_leaves(tree, matching_nodes):
    """ Recursively identifies nodes with exactly two leaves """
    if tree['feature'] is None:
        return
    if tree['left']['feature'] is None and tree['right']['feature'] is None:
        matching_nodes.append((tree['feature'], tree['value']))
        return matching_nodes
    find_nodes_with_two_leaves(tree['left'], matching_nodes)
    find_nodes_with_two_leaves(tree['right'], matching_nodes)
    return matching_nodes

def replace_node(tree, node, use_right=True):
    """ Replaces the specified node with its left or right child """
    node_feature, node_value = node
    if tree['feature'] is None:
        return
    if (tree['feature'] == node_feature and tree['value'] == node_value 
        and tree['left']['feature'] is None and tree['right']['feature'] is None):
        
        replacement = tree['right'] if use_right else tree['left']
        tree['feature'] = replacement['feature']
        tree['value'] = replacement['value']
        tree['left'] = replacement['left']
        tree['right'] = replacement['right']
        return
    replace_node(tree['left'], node, use_right)
    replace_node(tree['right'], node, use_right)
