import copy

from evaluate import simple_compute_accuracy


def prune_tree(tree, X_test, Y_test):
    """ Given a trained tree and test data, prune the tree to maximize accuracy
    """

    # Find all nodes directly connected to two leaves
    nodes_with_two_leaves = find_nodes_with_two_leaves(tree, [])
    nodes_visited = set()  # Store nodes already evaluated

    # Continue as long as there are candidates to prune
    while len(nodes_with_two_leaves) > 0:

        node = nodes_with_two_leaves.pop()  # Get node to evaluate
        # Skip if already checked, else add to visited set
        if node in nodes_visited:
            continue
        else:
            nodes_visited.add(node)

        # Copy tree so that pruning can be evaluated and reverted
        # Left tree will replace the node by the left child and
        # Right tree will replace the node by the right child
        tmp_tree_left = copy.deepcopy(tree)
        tmp_tree_right = copy.deepcopy(tree)

        # Replace the nodes
        replace_node(tmp_tree_left, node, False)
        replace_node(tmp_tree_right, node, True)

        # Compute baseline accuracy and left/right accuracy
        baseline_accuracy = simple_compute_accuracy(tree, X_test, Y_test)
        left_accuracy = simple_compute_accuracy(tmp_tree_left, X_test, Y_test)
        right_accuracy = simple_compute_accuracy(tmp_tree_right, X_test, Y_test)

        if left_accuracy >= baseline_accuracy or right_accuracy >= baseline_accuracy:

            # Left accuracy is the highest and therefore left tree is optimal
            if left_accuracy >= right_accuracy:
                # print(f"Left accuracy better for X[{node[0]}] > {node[1]}. {left_accuracy * 100}% vs {baseline_accuracy * 100}%")
                tree = tmp_tree_left
            else:
                # Right accuracy is the highest and therefore right tree is optimal
                # print(f"Right accuracy better for X[{node[0]}] > {node[1]}. {right_accuracy * 100}% vs {baseline_accuracy * 100}%")
                tree = tmp_tree_right

        # Check tree again and add any additional nodes with two leaves to the list to be evaluated
        new_nodes_with_two_leaves = find_nodes_with_two_leaves(tree, [])
        for new_node in new_nodes_with_two_leaves:
            if new_node in nodes_visited:
                continue
            else:
                nodes_with_two_leaves.append(new_node)

    return tree


def find_nodes_with_two_leaves(tree, matching_nodes):
    """ Given a tree, recursively find all nodes of the tree that are directly connected to
    two leaves
    """

    # Current node is a leaf so return
    if tree['feature'] is None:
        return

    # Current node is connected to two leaves
    if tree['left']['feature'] is None and tree['right']['feature'] is None:
        matching_nodes.append((tree['feature'], tree['value']))
        return matching_nodes

    # Recursively call on the left and right sub-trees
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

    # Current node is a match
    if tree['feature'] == node_feature and tree['value'] == node_value:
        if use_right:
            tree['value'] = tree['right']['value']
        else:
            tree['value'] = tree['left']['value']

        tree['feature'] = None
        tree['left'] = None
        tree['right'] = None

        return

    # Recursively call on left and right sub-trees
    replace_node(tree['left'], node, use_right)
    replace_node(tree['right'], node, use_right)
