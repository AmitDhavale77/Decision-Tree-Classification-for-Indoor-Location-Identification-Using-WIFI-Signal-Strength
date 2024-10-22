import numpy as np
from main import (
    calculate_entropy,
    calculate_information_gain,
    get_midpoints,
    split_data,
)
import evaluate as ev


def check_equal_output(expected_output, calculated_output):
    """
    Check if the expected output is equal to the calculated output
    """
    if np.all(expected_output == calculated_output):
        return "    passed test"
    else:
        return "    failed test"


def check_close_output(expected_output, calculated_output, error):
    """
    Check if the expected output is close to the calculated output
    """
    if abs(expected_output - calculated_output) <= error:
        return "    passed test"
    else:
        return "    failed test"


def test_entropy():
    dataset = np.array(
        [
            [200, 1],
            [78, 1],
            [90, 1],
            [400, 1],
            [356, 1],
            [98, 1],
            [149, 1],
            [79, 1],
            [156, 1],
            [180, 1],
            [788, 1],
            [543, 1],
            [976, 1],
            [655, 0],
            [345, 0],
            [444, 0],
            [754, 0],
            [43, 1],
            [33, 0],
            [21, 0],
            [34, 0],
            [39, 0],
            [41, 0],
            [45, 0],
            [27, 0],
            [18, 0],
            [22, 0],
            [49, 0],
            [22, 0],
            [24, 0],
        ]
    )
    truth_value = 0.996792
    Y = np.array([row[-1] for row in dataset])
    calculated_value = calculate_entropy(Y)
    error = 0.000001
    print("Testing entropy function:")
    return check_close_output(truth_value, calculated_value, error)


def test_ig():
    subset1 = np.array(
        [
            [200, 1],
            [78, 1],
            [90, 1],
            [400, 1],
            [356, 1],
            [98, 1],
            [149, 1],
            [79, 1],
            [156, 1],
            [180, 1],
            [788, 1],
            [543, 1],
            [976, 1],
            [655, 0],
            [345, 0],
            [444, 0],
            [754, 0],
        ]
    )
    subset2 = np.array(
        [
            [43, 1],
            [33, 0],
            [21, 0],
            [34, 0],
            [39, 0],
            [41, 0],
            [45, 0],
            [27, 0],
            [18, 0],
            [22, 0],
            [49, 0],
            [22, 0],
            [24, 0],
        ]
    )
    dataset = np.array(
        [
            [200, 1],
            [78, 1],
            [90, 1],
            [400, 1],
            [356, 1],
            [98, 1],
            [149, 1],
            [79, 1],
            [156, 1],
            [180, 1],
            [788, 1],
            [543, 1],
            [976, 1],
            [655, 0],
            [345, 0],
            [444, 0],
            [754, 0],
            [43, 1],
            [33, 0],
            [21, 0],
            [34, 0],
            [39, 0],
            [41, 0],
            [45, 0],
            [27, 0],
            [18, 0],
            [22, 0],
            [49, 0],
            [22, 0],
            [24, 0],
        ]
    )

    truth_value = 0.381215
    error = 0.000001
    Y = np.array([row[-1] for row in dataset])
    Y_left = np.array([row[-1] for row in subset1])
    Y_right = np.array([row[-1] for row in subset2])
    calculated_value = calculate_information_gain(Y, [Y_left, Y_right])
    print("Testing information gain function:")
    if abs(truth_value - calculated_value) <= error:
        return "    passed test"
    else:
        return "    failed test"


def test_get_midpoints():
    """
    Test the get_midpoints function
    """
    input = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
    expected_output = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    output = get_midpoints(input, 0)
    print("Testing midpoints function:")
    return check_equal_output(expected_output, output)


def test_split_data():
    """
    Test split data function
    """
    X = [
        [3, 5, 1, 8, 9, 9, 10],
        [11, 54, 3, 1, 8, 0, 34],
        [45, 0, 10, 6, 11, 13, 49],
        [22, 67, 21, 90, 4, 56, 7],
        [5, 3, 22, 7, 90, 88, 7],
        [8, 43, 30, 8, 88, 90, 1],
        [98, 101, 40, 87, 1, 2, 98],
        [43, 7, 54, 8, 11, 12, 13],
        [1, 2, 60, 3, 4, 5, 6],
        [7, 8, 76, 9, 10, 11, 12],
    ]
    Y = [1, 2, 3, 4, 5, 6, 7]
    X_left, Y_left, X_right, Y_right = split_data(X, Y, 2, 15.5)

    X_left_truth = [
        [3, 5, 1, 8, 9, 9, 10],
        [11, 54, 3, 1, 8, 0, 34],
        [45, 0, 10, 6, 11, 13, 49],
    ]

    Y_left_truth = [1, 2, 3]

    X_right_truth = [
        [22, 67, 21, 90, 4, 56, 7],
        [5, 3, 22, 7, 90, 88, 7],
        [8, 43, 30, 8, 88, 90, 1],
        [98, 101, 40, 87, 1, 2, 98],
        [43, 7, 54, 8, 11, 12, 13],
        [1, 2, 60, 3, 4, 5, 6],
        [7, 8, 76, 9, 10, 11, 12],
    ]

    Y_right_truth = [4, 5, 6, 7]
    print("Testing split data function:")
    if (
        X_left == X_left_truth
        and Y_left == Y_left_truth
        and X_right == X_right_truth
        and Y_right == Y_right_truth
    ):
        return "    passed test"
    else:
        return "    failed test"


def test_confusion_matrix():
    """
    Test confusion matrix function
    """
    y_true = np.array([2, 3, 0, 1, 3, 2, 1, 2, 3, 3])
    y_pred = np.array([2, 3, 0, 2, 3, 2, 0, 2, 3, 2])
    confusion_matrix = ev.get_confusion_matrix(y_true, y_pred)
    confusion_matrix_truth = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 3, 0], [0, 0, 1, 3]]
    )
    print("Testing confusion matrix function:")
    ev.show_confusion_matrix(confusion_matrix)
    return check_equal_output(confusion_matrix_truth, confusion_matrix)


def test_accuracy():
    """
    Test accuracy function
    """
    confusion_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 3, 0], [0, 0, 1, 3]]
    )
    accuracy_truth = 0.7
    print("Testing accuracy function:")
    accuracy = ev.compute_accuracy(confusion_matrix)
    return check_close_output(accuracy_truth, accuracy, 0.000001)


def test_recall():
    """
    Test recall function
    """
    confusion_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 3, 0], [0, 0, 1, 3]]
    )
    recall_truth = [1, 0, 1, 0.75]
    print("Testing recall function:")
    for index, true_recall in enumerate(recall_truth):
        recall = ev.compute_recall(confusion_matrix, index)
        recall_is_none = recall is None and true_recall is None
        assert recall_is_none or (abs(recall - true_recall) < 0.000001)
    return "    passed test"


def test_precision():
    """Test compute_precision function"""
    confusion_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 3, 0], [0, 0, 1, 3]]
    )
    precision_truth = [0.5, None, 0.6, 1]
    print("Testing precision function:")
    for index, true_precision in enumerate(precision_truth):
        precision = ev.compute_precision(confusion_matrix, index)
        precision_is_none = precision is None and true_precision is None
        assert precision_is_none or (abs(precision - true_precision) < 0.000001)
    return "    passed test"


def test_f1():
    """Test compute_f1 function"""
    recall_truth = [1, 0, 1, 0.75]
    precision_truth = [0.5, None, 0.6, 1]
    f1_truth = [0.6666666666666666, None, 0.7499999999999999, 0.8571428571428571]
    print("Testing f1 function:")
    for index, true_f1 in enumerate(f1_truth):
        f1 = ev.compute_f1(recall_truth[index], precision_truth[index])
        f1_is_none = f1 is None and true_f1 is None
        assert f1_is_none or (abs(f1 - true_f1) < 0.000001)
    return "    passed test"


def test_class_evaluation():
    """
    Test get_classification_evaluation function
    """
    confusion_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 3, 0], [0, 0, 1, 3]]
    )
    evaluation_truth_0 = {
        "precision": 0.5,
        "recall": 1,
        "f1": 0.66666666666666661,
    }
    print("Testing get_classification_evaluation function:")
    evaluation = ev.get_classification_evaluation(confusion_matrix, classification=0)
    for key in evaluation_truth_0:
        true_value = evaluation_truth_0[key]
        value = evaluation[key]
        value_is_none = value is None and true_value is None
        assert value_is_none or (abs(value - true_value) < 0.000001)
    return "    passed test"


def test_macro_average():
    """
    Test macro_average function
    """
    confusion_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 3, 0], [0, 0, 1, 3]]
    )
    macro_average_truth = {
        "precision": None,
        "recall": 0.6875,
        "f1": None,
    }
    evaluation_test = {
        "0": {"precision": 0.5, "recall": 1, "f1": 0.6666666666666666},
        "1": {"precision": None, "recall": 0, "f1": None},
        "2": {"precision": 0.6, "recall": 1, "f1": 0.7499999999999999},
        "3": {"precision": 1, "recall": 0.75, "f1": 0.8571428571428571},
    }
    print("Testing macro_average function:")
    macro_average = ev.compute_macroaverage(evaluation_test, ["0", "1", "2", "3"])
    for metric in macro_average_truth:
        true_value = macro_average_truth[metric]
        value = macro_average[metric]
        if value is None or true_value is None:
            value_is_none = value is None and true_value is None
            assert value_is_none, f"for {metric}: {value} != {true_value}"
        else:
            assert abs(value - true_value) < 0.000001
    return "    passed test"


if __name__ == "__main__":
    # Run all tests
    print(test_get_midpoints())
    print(test_entropy())
    print(test_ig())
    # print(test_split_data())
    print(test_confusion_matrix())
    print(test_accuracy())
    print(test_recall())
    print(test_precision())
    print(test_f1())
    print(test_class_evaluation())
    print(test_macro_average())
