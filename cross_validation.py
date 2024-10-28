import numpy as np
from decision_tree import decision_tree_learning
from evaluate import evaluation, compute_accuracy, compute_f1, compute_macroaverage
from prune import prune_tree


def nested_k_folds(X_data, Y_data, k=10, random_seed=60012):
    """
    Run nested k-folds cross validation on the dataset

    Args:
        X_data: features
        Y_data: labels
        k: number of folds
        random_generator: random number generator

    Returns:
        eval_list: list of dictionaries of evaluation metrics for each fold
    """
    # Create random generator
    random_generator = np.random.default_rng(random_seed)

    # Create array of shuffled indices
    shuffled_indices = random_generator.permutation(len(Y_data))
    split_indices = np.array_split(shuffled_indices, k)

    # Join features and labels
    if Y_data.ndim == 1:
        Y_data = Y_data.reshape(-1, 1)
    dataset = np.concatenate((X_data, Y_data), axis=1)

    # Iterate through the folds
    eval_list = []
    for i in range(len(split_indices)):
        # get indices for test split and train split
        test_split_ind = split_indices[i]
        train_indices = np.hstack(split_indices[:i] + split_indices[i + 1 :])
        # split data and run inner loop of the nested k-folds
        test_data = dataset[test_split_ind, :]
        train_eval_data = dataset[train_indices, :]
        eval_list.append(make_data_folds_new(train_eval_data, test_data))

    return eval_list


def make_data_folds_new(dataset, test_split, random_seed=60012, k=10):
    """
    TODO
    """
    random_generator = np.random.default_rng(random_seed)

    shuffled_indices = random_generator.permutation(len(dataset))
    data_shuffled = [dataset[i] for i in shuffled_indices]
    test_parts = np.array_split(data_shuffled, k, axis=0)

    labels = np.unique(dataset[:, -1])
    classes = [str(classification) for classification in labels]

    final_eval = {}
    for index, test_part in enumerate(test_parts):
        other_test_parts = [
            p for i, p in enumerate(test_parts) if i != index
        ]  # exclude current part by index

        # concatenate all other parts
        combined_data = np.vstack(other_test_parts)

        # separate features and labels
        X_train = combined_data[:, :-1]  # all columns except the last
        Y_train = combined_data[:, -1]  # last column

        decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)

        val_test_x = test_part[:, :-1]
        val_test_y = test_part[:, -1]

        prune_tree1 = prune_tree(decision_tree, val_test_x, val_test_y)

        X_test = test_split[:, :-1]
        Y_test = test_split[:, -1]

        evaluation_metrics = evaluation(X_test, Y_test, prune_tree1)

        # add the confusion matrix
        final_eval["confusion_matrix"] = (
            final_eval.get("confusion_matrix", 0)
            + evaluation_metrics["confusion_matrix"]
        )

        # calculate sum of metrics for exach class
        for index_str in classes:
            for metric in ["precision", "recall"]:
                final_entry = final_eval.get(index_str, {})

                # safely get the value or default to 0 if not present
                final_eval[index_str] = final_entry
                final_eval[index_str][metric] = final_entry.get(
                    metric, 0
                ) + evaluation_metrics[index_str].get(metric, 0)

    # calculate averages across all folds
    for index_str in classes:
        for metric in ["precision", "recall"]:
            final_eval[index_str][metric] /= k
        final_eval[index_str]["f1"] = compute_f1(
            final_eval[index_str]["precision"], final_eval[index_str]["recall"]
        )

    # calculate accuracy
    confusion_matrix1 = final_eval["confusion_matrix"]
    final_eval["accuracy"] = compute_accuracy(confusion_matrix1)

    # calculate macro-averaged overall metrics
    macro_avg = compute_macroaverage(final_eval, classes)
    final_eval["overall"] = {}
    for metric in macro_avg:
        final_eval["overall"][metric] = macro_avg[metric]

    return final_eval


def make_data_folds(dataset, random_seed, k=10):
    """
    TODO
    """
    # Create random generator
    random_generator = np.random.default_rng(random_seed)

    # Create array of shuffled indices
    shuffled_indices = random_generator.permutation(len(dataset))
    data_shuffled = [dataset[i] for i in shuffled_indices]
    test_parts = np.array_split(data_shuffled, k, axis=0)
    final_eval = {}
    labels = np.unique(dataset[:, -1])
    classes = [str(classification) for classification in labels]

    for index, test_part in enumerate(test_parts):
        other_test_parts = [
            p for i, p in enumerate(test_parts) if i != index
        ]  # exclude current part by index

        # concatenate all other parts
        combined_data = np.vstack(other_test_parts)

        # separate features and labels
        X_train = combined_data[:, :-1]  # all columns except the last
        Y_train = combined_data[:, -1]  # last column

        decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)

        evaluation_metrics = evaluation(
            test_part[:, :-1], test_part[:, -1], decision_tree
        )

        # add the confusion matrix
        final_eval["confusion_matrix"] = (
            final_eval.get("confusion_matrix", 0)
            + evaluation_metrics["confusion_matrix"]
        )

        # calculate sum of metrics for each class
        for index_str in classes:
            for metric in ["precision", "recall"]:
                final_entry = final_eval.get(index_str, {})

                # safely get the value or default to 0 if not present
                final_eval[index_str] = final_entry
                final_eval[index_str][metric] = final_entry.get(
                    metric, 0
                ) + evaluation_metrics[index_str].get(metric, 0)

    # calculate averages across all folds
    for index_str in classes:
        for metric in ["precision", "recall"]:
            final_eval[index_str][metric] /= k
        final_eval[index_str]["f1"] = compute_f1(
            final_eval[index_str]["precision"], final_eval[index_str]["recall"]
        )

    # calculate accuracy
    confusion_matrix1 = final_eval["confusion_matrix"]
    final_eval["accuracy"] = compute_accuracy(confusion_matrix1)

    # calculate macro-averaged overall metrics
    macro_avg = compute_macroaverage(final_eval, classes)
    final_eval["overall"] = {}
    for metric in macro_avg:
        final_eval["overall"][metric] = macro_avg[metric]

    return final_eval
