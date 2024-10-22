import numpy as np


# evaluation_example = {
#     '1': {
#         'accuracy': None,
#         'precision': None,
#         'recall': None,
#         'f1': None,
#     },
#     # etc... for 2, 3, 4
#     'confusion_matrix': None # 4x4 matrix
#     'overall': {
#         # same as above

#     }
# }


def compute_accuracy(y_gold, y_prediction):
    """Compute the accuracy given the ground truth and predictions

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
        row_predict[row] = predict(tree, test_row)

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
        return predict(tree["right"], test_row)
    else:
        return predict(tree["left"], test_row)


def get_classification_evaluation():
    """return dictionary in format:
    {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
    }
    """
    pass


def get_confusion_matrix(y_gold, y_prediction):
    """
    create the confusion matrix from the true and predicted values

    Args:
        y_gold: TODO
        y_prediction

    Returns:
        np.array of size n_classifications x n_classifications
    """
    # get number of unique classifications (assuming they all appear in the data)
    n_classifications = len(np.unique(y_gold))

    # create empty matrix
    confusion_matrix = np.zeros((n_classifications, n_classifications))

    # fill in confusion matrix
    for gold, prediction in zip(y_gold, y_prediction):
        confusion_matrix[gold, prediction] += 1

    return confusion_matrix


def show_confusion_matrix(confusion_matrix):
    """
    Print the confusion matrix in an easy to read format
    """
    print(f"Prediction class: {list(range(confusion_matrix.shape[0]))}")
    for classification, row in enumerate(confusion_matrix):
        print(f"True class {classification}:  {row}")


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
    # predict output
    x_test = test_db[:, :-1]
    y_gold = test_db[:, -1]
    y_prediction = predictions(trained_tree, x_test)

    # evaluate results
    confusion_matrix = get_confusion_matrix(y_gold, y_prediction)

    evaluation = {"confusion_matrix": confusion_matrix}

    classes = [str(classification) for classification in test_db[:, -1].unique()]
    for classification in classes:
        evaluation[classification] = get_classification_evaluation()

    # macroaverage overall metrics
    evaluation["overall"] = {}

    return evaluation


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
