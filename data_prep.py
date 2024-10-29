import numpy as np


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
    data = np.loadtxt(filename, dtype=np.float64)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


def train_test_split(X, Y, test_proportion, random_seed):
    """Split data X and Y into random train and test subsets according
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


def sort_data(X, Y, feature):
    """Sort data X, Y based on the feature column of X

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
    indices_sorted = np.argsort(dataset[:, feature])
    dataset_sorted = dataset[indices_sorted]

    # Separate dataset back into X and Y
    X_sorted = dataset_sorted[:, :-1]
    Y_sorted = dataset_sorted[:, -1]
    return X_sorted, Y_sorted


def split_data(X, Y, split_attribute, split_value):
    """This should take in results from above and acutally split the data

    Args:
        X : data
        Y : labels
        split_attribute (int) : column depeneding on which the split is done
        split_value (float) : value to split on (left split is values <= split_point and right split is values > split_point)

    Returns:
        X_left, Y_left, X_right, Y_right
    """
    ind_mid = X[:, split_attribute] > split_value
    X_right = X[ind_mid]
    X_left = X[~ind_mid]

    Y_right = Y[ind_mid]
    Y_left = Y[~ind_mid]

    return X_left, Y_left, X_right, Y_right
