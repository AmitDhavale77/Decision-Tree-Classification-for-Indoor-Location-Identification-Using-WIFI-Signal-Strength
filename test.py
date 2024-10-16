import numpy as np
from main import calculate_entropy, calculate_information_gain

def test_entropy():
    dataset = np.array([[200, 1],[78, 1],[90, 1],[400, 1],[356, 1],[98, 1],[149, 1],[79, 1], [156, 1], [180, 1], [788, 1], [543, 1],[976, 1], [655, 0], [345, 0], [444, 0], [754, 0], [43, 1], [33, 0], [21, 0], [34, 0], [39, 0], [41, 0], [45, 0], [27, 0], [18, 0], [22, 0], [49, 0], [22, 0], [24, 0] ])
    truth_value = 0.996792
    calculated_value = calculate_entropy(dataset)
    error = 0.000001
    if abs(truth_value - calculated_value) <= error:
        return "passed test"
    else:
        return "failed test"

def test_ig():
    subset1 = np.array([[200, 1],[78, 1],[90, 1],[400, 1],[356, 1],[98, 1],[149, 1],[79, 1], [156, 1], [180, 1], [788, 1], [543, 1],[976, 1], [655, 0], [345, 0], [444, 0], [754, 0]])
    subset2 = np.array([[43, 1], [33, 0], [21, 0], [34, 0], [39, 0], [41, 0], [45, 0], [27, 0], [18, 0], [22, 0], [49, 0], [22, 0], [24, 0]])
    dataset = np.array([[200, 1],[78, 1],[90, 1],[400, 1],[356, 1],[98, 1],[149, 1],[79, 1], [156, 1], [180, 1], [788, 1], [543, 1],[976, 1], [655, 0], [345, 0], [444, 0], [754, 0], [43, 1], [33, 0], [21, 0], [34, 0], [39, 0], [41, 0], [45, 0], [27, 0], [18, 0], [22, 0], [49, 0], [22, 0], [24, 0] ])

    truth_value = 0.381215
    error = 0.000001
    calculated_value = calculate_information_gain(dataset, [subset1, subset2])
    if abs(truth_value - calculated_value) <= error:
        return "passed test"
    else:
        return "failed test"

print(test_entropy())
print(test_ig())
