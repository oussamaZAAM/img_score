import numpy as np


def get_score(prediction):
    # Example input array of numbers
    input_array = np.array(prediction)

    # Example input arrays
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = np.array(input_array)

    score = np.sum(a * b) / np.sum(b)

    return score
