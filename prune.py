import copy  # part of python standard library
from evaluate import simple_compute_accuracy


def prune_tree(tree, X_test, Y_test):
    """ Given a trained tree and test data, prune the tree to maximize accuracy
    """

    # Find all nodes directly connected to two leaves
    nodes_with_two_leaves = find_nodes_with_two_leaves(tree, [])
    nodes_visited = set()  # Store nodes already evaluated
    baseline_accuracy = simple_compute_accuracy(tree, X_test, Y_test)

    while nodes_with_two_leaves:
        node = nodes_with_two_leaves.pop()  # Get node to evaluate

        if node in nodes_visited:
            continue
        nodes_visited.add(node)

        # Copy tree so that pruning can be evaluated and reverted
        # Left tree will replace the node by the left child and
        # Right tree will replace the node by the right child
        tmp_tree_left = copy.deepcopy(tree)
        tmp_tree_right = copy.deepcopy(tree)
        replace_node(tmp_tree_left, node, use_right=False)
        replace_node(tmp_tree_right, node, use_right=True)

        # Calculate accuracies for pruned versions
        left_accuracy = simple_compute_accuracy(tmp_tree_left, X_test, Y_test)
        right_accuracy = simple_compute_accuracy(tmp_tree_right, X_test, Y_test)

        # Update tree if accuracy improves
        if left_accuracy >= baseline_accuracy or right_accuracy >= baseline_accuracy:

            # Left accuracy is the highest and therefore left tree is optimal
            if left_accuracy >= right_accuracy:
                tree = tmp_tree_left
                baseline_accuracy = left_accuracy
            else:
                # Right accuracy is the highest and therefore right tree is optimal
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
    """ Given a tree and a node, find that node in the tree and replace it with
    it's child. If use_right, replace with the right child, else replace with left child
    """

    node_feature, node_value = node

    # Current node is a leaf
    if tree['feature'] is None:
        return
    
    # Current node is a match and validate that it is connected to two leaves
    if (tree['feature'] == node_feature and tree['value'] == node_value 
        and tree['left']['feature'] is None and tree['right']['feature'] is None):
        
        replacement = tree['right'] if use_right else tree['left']
        tree['feature'] = replacement['feature']
        tree['value'] = replacement['value']
        tree['left'] = replacement['left']
        tree['right'] = replacement['right']
        return

    # Recursively call on left and right sub-trees
    replace_node(tree['left'], node, use_right)
    replace_node(tree['right'], node, use_right)
