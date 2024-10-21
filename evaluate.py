# Do evaluation stuff


evaluation_example = {
    '1': {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
    },
    # etc... for 2, 3, 4
    'confusion_matrix': None # 4x4 matrix
    'overall': {
        # same as above

    }
}


def compute_accuracy(y_gold, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        accuracy (float): accuracy value between 0 and 1
    """

    assert len(y_gold) == len(y_prediction)

    if len(y_gold) == 0:
        return 0

    return np.sum(y_gold == y_prediction) / len(y_gold)


def evaluate_tree(tree, test_x):
    """ Evaluate accuracy of trained tree using test data
    @aanish
    """

    num_rows, _ = test_x.shape
    predictions = np.zeros((num_rows,))

    for row in range(num_rows):
        test_row = test_x[row, :]
        predictions[row] = predict(tree, test_row)

    return predictions


def predict(tree, test_row):
    """ Make a prediction on an input dataset using a trained tree
    """

    if tree['feature'] is None:
        return tree['value']

    cur_feature = tree['feature']
    cur_value = tree['value']
    go_right = test_row[cur_feature] > cur_value
    if go_right:
        return predict(tree['right'], test_row)
    else:
        return predict(tree['left'], test_row)


def evaluation(test_db, trained_tree):
    """
    @Lauren
    Confusion Matrix (4x4 matrix)
    Then for each label:
        Accuracy
        Precision/Recall
        F1 score
    Then do macro-averaging to get overall
    Should we do micro-averaging??? TBD

    Return a dictionary with all of these metrics
    """

    # Ideally this should return a single metric we can make decision on
    # Do the math
    return evaluation_example


def advanced_k_fold(dataset, k=10):
    # @Amit
    # implement option 2 from slide 30 for cross-validation 
    # this will be used for parameter tuning the pruning part

    # this will call make_data_folds

    # Return evaluation dictionary in the same format as above one but
    # averaged across the k folds

    # this will use prune_tree to test on different trees
    return


def make_data_folds(dataset, k=10):
    # @Ola
    # Look at existing version of train_test_split but will need to make folds deterministically

    # Shuffle data
    # Then pick first two rows for test, rest is training+validation
    # Evaluate that data
    # then pick next two rows, cont...
    # As you go through, keep track of averaged evaluation metrics (all of them)

    # Return evaluation dictionary in the same format as above one but
    # averaged across the k folds
    return
