import numpy as np


def simple_compute_accuracy(y_gold, y_prediction):
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
        predictions[row] = row_predict(tree, test_row)

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
        return row_predict(tree['right'], test_row)
    else:
        return row_predict(tree['left'], test_row)


def compute_accuracy(confusion_matrix):
    """Compute the accuracy given the ground truth and predictions

    Args:
        confusion_matrix (np.array): confusion matrix of size n_classifications x n_classifications

    Returns:
        accuracy (float): accuracy value between 0 and 1
    """
    class_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return class_accuracy


def compute_precision(confusion_matrix, classification):
    """
    Calculate the precision for a single classification

    Args:

    Returns: float
    """
    denominator = np.sum(confusion_matrix[:, classification])
    if denominator == 0:
        return None
    else:
        return confusion_matrix[classification, classification] / denominator


def compute_recall(confusion_matrix, classification):
    """ """
    denominator = np.sum(confusion_matrix[classification, :])
    if denominator == 0:
        return None
    else:
        return confusion_matrix[classification, classification] / denominator


def compute_f1(precision, recall):
    """"""
    if precision is None or recall is None:
        return None
    else:
        return 2 * (precision * recall) / (precision + recall)


def get_classification_evaluation(confusion_matrix, classification):
    """
    Args: TODO

    Return: dict:  in format:
            {
                'precision': None,
                'recall': None,
                'f1': None,
            }
    """
    precision = compute_precision(confusion_matrix, classification)
    recall = compute_recall(confusion_matrix, classification)
    f1 = compute_f1(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_confusion_matrix(y_gold, y_prediction):
    """
    create the confusion matrix from the true and predicted values

    Args:
        y_gold: TODO
        y_prediction

    Returns:
        np.array of size n_classifications x n_classifications
    """
    # create empty matrix
    n_classifications = len(np.unique(y_gold))
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


def compute_macroaverage(evaluation, classes):
    """ """
    macroaverage = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }
    assert isinstance(evaluation, dict)
    for classification in classes:
        for metric in macroaverage:
            if (
                evaluation[classification][metric] is not None
                and macroaverage[metric] is not None
            ):
                macroaverage[metric] += evaluation[classification][metric]
            else:
                macroaverage[metric] = None

    for metric in macroaverage:
        if macroaverage[metric] is not None:
            macroaverage[metric] = macroaverage[metric] / len(classes)

    return macroaverage


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
    confusion_matrix = get_confusion_matrix(y_gold, y_prediction)

    # evaluate results
    evaluation = {}
    evaluation["confusion_matrix"] = confusion_matrix
    evaluation["accuracy"] = compute_accuracy(confusion_matrix)

    # evaluate each classification
    classes = [str(classification) for classification in np.unique(test_db[:, -1])]
    for classification in classes:
        evaluation[classification] = get_classification_evaluation(
            confusion_matrix, int(classification)
        )

    # macroaverage overall metrics
    evaluation["overall"] = compute_macroaverage(evaluation, classes)

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
