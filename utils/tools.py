import numpy as np
from scipy.stats import ttest_ind
from utils.metrics import compute_accuracy, compute_msg

def evaluate_results(results):
    """
    Process predictions stored in memory to compute accuracy, MSG, and perform T-tests.

    Args:
        results (dict): A dictionary where keys are model-test set combinations,
                        and values are lists of (label, prediction) tuples.

    Returns:
        None: Prints accuracy, MSG, and T-test results.
    """
    # Compute accuracies for each model-test set combination
    accuracies = {}
    for key, predict_log in results.items():
        accuracies[key] = compute_accuracy(predict_log)

    # Print accuracies
    for key, acc in accuracies.items():
        print(f"Accuracy for {key}: {acc:.4f}")

    # Extract results for Model Mixed
    model_mixed_test_ell = results["model_mixed_test_ell"]
    model_mixed_test_non_ell = results["model_mixed_test_non_ell"]
    model_ell_test_ell = results["model_ell_test_ell"]
    model_ell_test_non_ell = results["model_ell_test_non_ell"]
    model_non_ell_test_ell = results["model_non_ell_test_ell"]
    model_non_ell_test_non_ell = results["model_non_ell_test_non_ell"]

    # MSG computation
    msg_mixed_model_ell_vs_non_ell= compute_msg(model_mixed_test_ell, model_mixed_test_non_ell, mode="compare")
    msg_ell_model_ell_vs_non_ell = compute_msg(model_ell_test_ell, model_ell_test_non_ell, mode="compare")
    msg_non_ell_model_ell_vs_non_ell = compute_msg(model_non_ell_test_ell, model_non_ell_test_non_ell, mode="compare")
    
    msg_mixed_model_ell_vs_human = compute_msg(model_mixed_test_ell, mode="human")
    msg_mixed_model_non_ell_vs_human = compute_msg(model_mixed_test_non_ell, mode="human")
    msg_ell_model_ell_vs_human = compute_msg(model_ell_test_ell, mode="human")
    msg_ell_model_non_ell_vs_human = compute_msg(model_ell_test_non_ell, mode="human")
    msg_non_ell_model_ell_vs_human = compute_msg(model_non_ell_test_ell, mode="human")
    msg_non_ell_model_non_ell_vs_human = compute_msg(model_non_ell_test_non_ell, mode="human")
    

    print(f"MSG (Mixed model: ELL vs Non-ELL): {msg_mixed_model_ell_vs_non_ell:.4f}")
    print(f"MSG (ELL model: ELL vs Non-ELL): {msg_ell_model_ell_vs_non_ell:.4f}")
    print(f"MSG (Non-ELL model: ELL vs Non-ELL): {msg_non_ell_model_ell_vs_non_ell:.4f}")

    print(f"MSG (Mixed model: Human vs ELL): {msg_mixed_model_ell_vs_human:.4f}")
    print(f"MSG (Mixed model: Human vs Non-ELL): {msg_mixed_model_non_ell_vs_human:.4f}")
    print(f"MSG (ELL model: Human vs ELL): {msg_ell_model_ell_vs_human:.4f}")
    print(f"MSG (ELL model: Human vs Non-ELL): {msg_ell_model_non_ell_vs_human:.4f}")
    print(f"MSG (Non-ELL model: Human vs ELL): {msg_non_ell_model_ell_vs_human:.4f}")
    print(f"MSG (Non-ELL model: Human vs Non-ELL): {msg_non_ell_model_non_ell_vs_human:.4f}")