import numpy as np

def compute_accuracy(predict_log):
    """
    Compute the accuracy from a list of label-prediction pairs.

    Args:
        predict_log (list): A list of tuples, where each tuple contains the true label and the predicted label.

    Returns:
        float: The accuracy calculated as the proportion of correct predictions.
               Returns 0.0 if the input list is empty.
    """
    accuracies = []
    if not predict_log:
        return 0.0
    correct_counts = sum(1 for (label, pred) in predict_log if label == pred)
    total_counts = len(predict_log)
    return correct_counts / total_counts


def compute_msg(data_a, data_b=None, col_index=1, mode="compare"):
    """
    Compute Mean Score Gap (MSG) based on the specified mode.

    Args:
        data_a (list or np.ndarray): The first dataset (required).
        data_b (list or np.ndarray): The second dataset (optional, required for 'compare' mode).
        col_index (int): The column index to compute mean for (default: 1).
        mode (str): The computation mode, either 'human' or 'compare':
            - 'human': Compute the absolute difference between the means of two columns in the same dataset.
            - 'compare': Compute the absolute difference of column means between two datasets.

    Returns:
        float: The computed MSG value.
    """
    data_a = np.array(data_a)

    if mode == "human":
        # Compute mean of both columns in a single dataset
        mean_col0 = np.mean(data_a[:, 0])
        mean_col1 = np.mean(data_a[:, 1])
        msg = abs(mean_col1 - mean_col0)
        return msg

    elif mode == "compare":
        if data_b is None:
            raise ValueError("data_b is required for 'compare' mode.")
        data_b = np.array(data_b)
        # Compute mean of the specified column for both datasets
        mean_a = np.mean(data_a[:, col_index])
        mean_b = np.mean(data_b[:, col_index])
        msg = abs(mean_a - mean_b)
        return msg

    else:
        raise ValueError("Invalid mode. Choose either 'human' or 'compare'.")