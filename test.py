import numpy as np
from main import calculate_entropy, calculate_information_gain, get_midpoints


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
    calculated_value = calculate_entropy(dataset)
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
    calculated_value = calculate_information_gain(dataset, [subset1, subset2])
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


if __name__ == "__main__":
    # Run all tests
    print(test_get_midpoints())
    print(test_entropy())
    print(test_ig())
