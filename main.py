import numpy as np

from evaluate import evaluation, compute_accuracy, compute_macroaverage, compute_f1, simple_compute_accuracy
from prune import prune_tree
from visualize import visualize_tree, tree_to_json, json_to_tree


def load_data(filename):
    """Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes corresponding to WIFI
                   signals from different receivers
               - y is a numpy array with shape (N, ), and each element should be
                   an integer representing room number from 0 to C-1 where C
                   is the number of classes
    """

    X = np.loadtxt(filename, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5, 6))
    Y = np.loadtxt(filename, usecols=(7)).astype(np.int64)
    return X, Y


def train_test_split(X, Y, test_proportion, random_seed):
    """ Split data X and Y into random train and test subsets according
    to specified test_size proportion

    Args:
        X (np.array): Feature data
        Y (np.array): Label data
        test_proportion (float): Indicates proportion of dataset to be
            test data. Must be between 0.0 and 1.0
        random_seed (int): Seed used to initialize random generator

    Returns:
        - X_train (np.array): X data to use for train
        - Y_train (np.array): Y data to use for train
        - X_test (np.array): X data to use for test
        - Y_test (np.array): Y data to use for test
    """

    # Create random generator
    random_generator = np.random.default_rng(random_seed)

    # Create array of shuffled indices
    shuffled_indices = random_generator.permutation(len(X))

    # Define size of test and train arrays based on input proportion
    num_test = round(len(X) * test_proportion)

    X_test = X[shuffled_indices[:num_test]]
    Y_test = Y[shuffled_indices[:num_test]]

    X_train = X[shuffled_indices[num_test:]]
    Y_train = Y[shuffled_indices[num_test:]]

    return X_train, Y_train, X_test, Y_test


def make_data_folds(dataset, random_seed, k=10):
    """
    """
    # Look at existing version of train_test_split but will need to make folds deterministically

    # Shuffle data
    # Then pick first two rows for test, rest is training+validation
    # Evaluate that data
    # then pick next two rows, cont...
    # As you go through, keep track of averaged evaluation metrics (all of them)

    # Return evaluation dictionary in the same format as above one but
    # averaged across the k folds
    # Create random generator


    random_generator = np.random.default_rng(random_seed)

    # Create array of shuffled indices
    shuffled_indices = random_generator.permutation(len(dataset))


    data_shuffled = [dataset[i] for i in shuffled_indices] 

    test_parts = np.array_split(data_shuffled, k, axis=0)

    final_eval = {}

    labels = np.unique(dataset[:,-1])

    classes = [str(classification) for classification in labels]

    for index, test_part in enumerate(test_parts):
        other_test_parts = [p for i, p in enumerate(test_parts) if i != index]  # exclude current part by index

        # concatenate all other parts
        combined_data = np.vstack(other_test_parts)  

        # separate features and labels 
        X_train = combined_data[:, :-1]  # all columns except the last
        Y_train = combined_data[:, -1]   # last column

        decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)

        evaluation_metrics = evaluation(test_part[:,:-1], test_part[:,-1], decision_tree)

        # add the confusion matrix
        final_eval['confusion_matrix'] = final_eval.get('confusion_matrix', 0) + evaluation_metrics['confusion_matrix']

        # calculate sum of metrics for exach class
        for index_str in classes:
            for metric in ['precision', 'recall']:
                final_entry = final_eval.get(index_str, {})

                # safely get the value or default to 0 if not present
                final_eval[index_str] = final_entry  
                final_eval[index_str][metric] = final_entry.get(metric, 0) + evaluation_metrics[index_str].get(metric, 0)

    # calculate averages across all folds
    for index_str in classes:
        for metric in ['precision', 'recall']:
            final_eval[index_str][metric] /= k
        final_eval[index_str]['f1'] = compute_f1(final_eval[index_str]['precision'], final_eval[index_str]['recall'])

    # calculate accuracy
    confusion_matrix1 = final_eval['confusion_matrix']
    final_eval['accuracy'] = compute_accuracy(confusion_matrix1)


    # calculate macro-averaged overall metrics
    macro_avg = compute_macroaverage(final_eval, classes)
    final_eval['overall'] = {}
    for metric in macro_avg :
        final_eval['overall'][metric] = macro_avg[metric]


    return final_eval


def create_node(feature, value, left, right):
    """ Given node attributes, instantiate the node as a dictionary
    """

    node = {
        'feature': feature,
        'value': value,
        'left': left,
        'right': right,
    }

    return node


def calculate_entropy(data):
    """ Calculate the entropy of a given array

    data : np.array with 1 dimension
    """

    if len(data) == 1:
        return 0  # if there is only one instance then entropy is 0
    else:
        dict_classes = {}  # dictionary counting how many elements are in particular classes
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
    """ Calculate the information gain of comparing Y with subsets (Yleft, Yright)

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
    """ Given input dataset X and Y, determine the optimal way to split the data
    by iteratively calculating information gain and finding the maximum

    Args: X, Y

    Returns: best_feature, best_value
    """

    # calculate information gain at each potential split point for each feature
    max_info_gain = 0
    best_feature = None
    best_value = None

    # TODO: confirm this?
    for feature in range(X.shape[1]):

        # Sort X by feature
        X_sorted, Y_sorted = sort_data(X, Y, feature)

        # Get midpoints of the feature column of sorted X
        midpoints = get_midpoints(X_sorted, feature)
        for midpoint in midpoints:
            # Split data into two groups based on split point
            _, Y_left, _, Y_right = split_data(
                X_sorted, Y_sorted, feature, midpoint
            )

            # Calculate information gain for each split point (feature data < value)
            split_point_ig = calculate_information_gain(
                Y_sorted, (Y_left, Y_right)
            )

            # If new split point is larger than the previous maximum, set it to be
            # the current best split option
            if split_point_ig > max_info_gain:
                max_info_gain = split_point_ig
                best_feature = feature
                best_value = midpoint

    return best_feature, best_value


def sort_data(X, Y, feature):
    """ Sort data X, Y based on the feature column of X

    Args:
        X (np.array): Feature data
        Y (np.array): Label data
        feature (int): Index of feature column to sort on

    Returns:
        - X_sorted (np.array): X sorted on X[feature]
        - Y_sorted (np.array): Y sorted on X[feature]
    """

    # Ensure feature is a valid column
    if feature not in range(X.shape[1]):
        raise ValueError(f"Feature {feature} does not exist in X")

    # Reshape Y so that it can be joined with X
    Y = Y.reshape(-1, 1)
    # Combine Y with X so that it gets sorted alongside it
    dataset = np.concatenate((X, Y), axis=1)

    # Use argsort to sort data on feature
    indices_sorted = np.argsort(dataset[:,feature])
    dataset_sorted = dataset[indices_sorted]

    # Separate dataset back into X and Y
    X_sorted = dataset_sorted[:, :-1]
    Y_sorted = dataset_sorted[:, -1]
    return X_sorted, Y_sorted


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


def split_data(X, Y, split_attribute, split_value):
    """ This should take in results from above and acutally split the data

    Args: 
        X : data
        Y : labels
        split_attribute (int) : column depeneding on which the split is done
        split_value (float) : value to split on (left split is values <= split_point and right split is values > split_point)

    Returns: 
        X_left, Y_left, X_right, Y_right
    """

    ind_mid = X[:,split_attribute] > split_value
    X_right = X[ind_mid]
    X_left = X[~ind_mid]

    Y_right = Y[ind_mid]
    Y_left = Y[~ind_mid]

    return X_left, Y_left, X_right, Y_right 


def check_all_elements_same(arr):
    """ Check if all elements in a numpy array are the same
    """

    if np.all(arr == arr[0]):
        return True, arr[0]
    else:
        return False, None


def decision_tree_learning(X, Y, depth=0, max_depth=None):
    """ Train the decision tree on dataset X and Y

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
    left_branch, left_depth = decision_tree_learning(X_left, Y_left, depth + 1, max_depth=max_depth)
    right_branch, right_depth = decision_tree_learning(X_right, Y_right, depth + 1, max_depth=max_depth)

    # Create node with current split feature and value
    node = create_node(split_feature, split_value, left_branch, right_branch)

    return node, max(left_depth, right_depth)


def count_leaves(tree):
    """ Calculate number of leaves i.e max span of the tree
    """

    if tree is None:
        return 0
    elif tree['feature'] is None:
        return 1

    return count_leaves(tree['left']) + count_leaves(tree['right'])


def get_tree_depth(tree):
    """ Calculate tree depth
    """

    if tree is None or tree['feature'] is None:
        return 0
    return 1 + max(get_tree_depth(tree['left']), get_tree_depth(tree['right']))


if __name__ == "__main__":
    clean_filename = 'wifi_db/clean_dataset.txt'
    noisy_filename = 'wifi_db/noisy_dataset.txt'


    X, Y = load_data(noisy_filename)

    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, 0.6, 42)
    # X_test, Y_test, X_cv, Y_cv = train_test_split(X_, Y_, 0.5, 42)

    decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)
    og_depth = get_tree_depth(decision_tree)
    og_span = count_leaves(decision_tree)
    og_accuracy = simple_compute_accuracy(decision_tree, X_test, Y_test)
    print(f"Original Depth: {og_depth} & Span: {og_span}")
    print(f"Original Accuracy: {og_accuracy}")

    # visualize_tree(decision_tree, og_depth)
    eval_dict = evaluation(X_test, Y_test, decision_tree)
    # tree_to_json(decision_tree, 'tree.json')
    # decision_tree = json_to_tree('noisy_tree.json')
    # visualize_tree(decision_tree, 11)
    # print(f"Accuracy on test set: {accuracy*100}%")
    #print(eval_dict)


    pruned_tree = prune_tree(decision_tree, X_test, Y_test)
    new_depth = get_tree_depth(pruned_tree)
    new_span = count_leaves(pruned_tree)
    new_accuracy = simple_compute_accuracy(pruned_tree, X_test, Y_test)

    print(f"Pruned Depth: {new_depth} & Span: {new_span}")
    print(f"New Accuracy: {new_accuracy}")

    data = np.loadtxt(noisy_filename)
    print(make_data_folds(data, 42, k=10))

    # visualize_tree(pruned_tree, new_depth)
