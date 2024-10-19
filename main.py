import numpy as np

from visualize import visualize_tree


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


def create_node(feature, value, left, right):
    """
    @amit - convert to use class for node
    """

    node = {
        'feature': feature,
        'value': value,
        'left': left,
        'right': right,
    }

    return node


def calculate_entropy(data):
    """
    data : np.array with 1 dimension
    @ola
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
    """
    @ola - related to above
    see if integral option is better (we think no but worth checking?)

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
                print(f"Better info gain! {split_point_ig} > {max_info_gain} on feature {feature} with value {midpoint}")
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

    midpoints = list(set(midpoints))  # remove duplicates
    return np.array(midpoints)


def split_data(X, Y, split_attribute, split_value):
    """This should take in results from above and acutally split the data
    @amit

    Args: 
        X : data
        Y : labels
        aplit_attribute (int) : column depeneding on which the split is done
        split_value (float) : value to split on (left split is values <= split_point and right split is values > split_point)

        data - TODO decide on data structure
       
    Returns: 
        X_left, Y_left, X_right, Y_right
    """
    flag = True
    index = 0
    while flag:
        if X[index][split_attribute] <= split_value:
            index += 1
        else:
            flag = False
    X_left = X[:index]
    Y_left = Y[:index]
    X_right = X[index:]
    Y_right = Y[index:]

    return X_left, Y_left, X_right, Y_right 


def check_all_elements_same(arr):
    """ Check if all elements in a numpy array are the same
    """

    if np.all(arr == arr[0]):
        return True, arr[0]
    else:
        return False, None


def decision_tree_learning(X, Y, depth=0, max_depth=None):
    """This code is a placeholder
    Once we do the other parts, can tackle this bit

    Args:
        X train
        Y train,
        depth, current depth
        max depth = how far should we go?

    Returns:
        tree (nested dictionary representing entire tree and branches and leafs)
    """

    # End decision tree learning if max depth has been met
    if depth == max_depth:
        return

    print(f"At depth: {depth}")

    # Check if all Y has the same label and if so create a leaf node
    y_has_same_label, label_value = check_all_elements_same(Y)
    if y_has_same_label:
        return create_node(
            None, label_value, None, None,
        ), depth  # Set leaf value to the label of Y

    split_attribute, split_value = find_split(X, Y)
    print(f"Optimal split found on feature {split_attribute} > {split_value}")
    X_left, Y_left, X_right, Y_right = split_data(X, Y, split_attribute, split_value)

    left_branch, left_depth = decision_tree_learning(X_left, Y_left, depth + 1, max_depth=max_depth)
    right_branch, right_depth = decision_tree_learning(X_right, Y_right, depth + 1, max_depth=max_depth)

    node = create_node(split_attribute, split_value, left_branch, right_branch)

    return node, max(left_depth, right_depth)


def evaluate_tree(test_data, trained_tree):
    """Evaluate accuracy of trained tree using test data
    @aanish
    """

    pass


def prune_tree():
    """wait till next week's lecture to see how to implement this"""

    pass


if __name__ == "__main__":
    clean_filename = 'wifi_db/clean_dataset.txt'
    noisy_filename = 'wifi_db/noisy_dataset.txt'

    X, Y = load_data(clean_filename)

    X_train, Y_train, X_, Y_ = train_test_split(X, Y, 0.2, 42)
    X_test, Y_test, X_cv, Y_cv = train_test_split(X_, Y_, 0.5, 42)

    decision_tree, depth_reached = decision_tree_learning(X_train, Y_train)
    print(depth_reached)
    visualize_tree(decision_tree)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_cv.shape)

    # print(X_cv[:, 5])
    # X_sorted, Y_sorted = sort_data(X_cv, Y_cv, 5)

    # print(X_sorted[:, 5])
    # print(get_midpoints(X_sorted, feature=5))

    # print((X_sorted[:, 5][1:] + X_sorted[:, 5][:-1])/2)
