import numpy as np
from main import calculate_entropy, calculate_information_gain, get_midpoints, split_data


def check_equal_output(expected_output, calculated_output):
    """
    Check if the expected output is equal to the calculated output
    """
    if all(expected_output == calculated_output):
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
        [7, 8, 76, 9, 10, 11, 12]
        ]
    Y = [1, 2, 3, 4, 5, 6, 7]
    X_left, Y_left, X_right, Y_right = split_data(X, Y, 2, 15.5)
    
    X_left_truth = [
    [3, 5, 1, 8, 9, 9, 10],
    [11, 54, 3, 1, 8, 0, 34],
    [45, 0, 10, 6, 11, 13, 49]
    ]

    Y_left_truth = [1, 2, 3]

    X_right_truth = [
    [22, 67, 21, 90, 4, 56, 7],
    [5, 3, 22, 7, 90, 88, 7],
    [8, 43, 30, 8, 88, 90, 1],
    [98, 101, 40, 87, 1, 2, 98],
    [43, 7, 54, 8, 11, 12, 13],
    [1, 2, 60, 3, 4, 5, 6],
    [7, 8, 76, 9, 10, 11, 12]
    ]

    Y_right_truth = [4, 5, 6, 7]
    print("Testing split data function:")
    if X_left == X_left_truth and Y_left == Y_left_truth and X_right == X_right_truth and Y_right == Y_right_truth:
        return "    passed test"
    else:
        return "    failed test"


if __name__ == "__main__":
    # Run all tests
    print(test_get_midpoints())
    print(test_entropy())
    print(test_ig())
    print(test_split_data())
