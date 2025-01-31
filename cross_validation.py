import numpy as np
from decision_tree import decision_tree_learning
from evaluate import evaluation, compute_accuracy, compute_f1, compute_macroaverage
from prune import prune_tree
from decision_tree import get_tree_depth, count_leaves


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
    pre_pruned_depths = []
    post_pruned_depths = []
    pre_pruned_spans = []
    post_pruned_spans = []
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
        output = make_data_folds_pruned(train_eval_data, test_data, k)
        eval_list.append(output[0])
        pre_pruned_depths.extend(output[1])
        post_pruned_depths.extend(output[2])
        pre_pruned_spans.extend(output[3])
        post_pruned_spans.extend(output[4])

    mean_pre_pruned_depths = np.mean(pre_pruned_depths)
    mean_post_pruned_depths = np.mean(post_pruned_depths)
    mean_pre_pruned_spans = np.mean(pre_pruned_spans)
    mean_post_pruned_spans = np.mean(post_pruned_spans)


    return eval_list, mean_pre_pruned_depths, mean_post_pruned_depths, mean_pre_pruned_spans, mean_post_pruned_spans


def make_data_folds_pruned(dataset, test_split, random_seed=60012, k=10):
    """
    Inner loop of the nested k-folds cross validation, creates tree, prunes 
    using validation data and evaluates on test data

    Args:
        dataset: training and validation data
        test_split: test data
        random_seed: random seed
        k: number of folds
    """
    pre_pruned_depths = []
    post_pruned_depths = []
    pre_pruned_spans = []
    post_pruned_spans = []
    random_generator = np.random.default_rng(random_seed)

    # creating k folds from the train-val data (dataset)
    shuffled_indices = random_generator.permutation(len(dataset))
    data_shuffled = [dataset[i] for i in shuffled_indices]
    train_val_folds = np.array_split(data_shuffled, k, axis=0)

    # create list of class labels
    labels = np.unique(dataset[:, -1])
    classes = [str(classification) for classification in labels]

    # iterating through each train-val fold
    final_eval = {}
    for index, val_data in enumerate(train_val_folds):
        # select data for training, excluding the current validation fold
        train_data = [p for i, p in enumerate(train_val_folds) if i != index]

        # concatenate all other parts
        combined_train_data = np.vstack(train_data)

        # separate features and labels for train data and train tree
        X_train = combined_train_data[:, :-1]  # all columns except the last
        Y_train = combined_train_data[:, -1]  # last column
        decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)
        pre_pruned_span = count_leaves(decision_tree)
        pre_pruned_depths.append(tree_depth)
        pre_pruned_spans.append(pre_pruned_span)

        # seperate features and labels for validation data and prune tree
        val_x = val_data[:, :-1]
        val_y = val_data[:, -1]
        pruned_tree = prune_tree(decision_tree, val_x, val_y)
        post_prune_depth = get_tree_depth(pruned_tree)
        post_pruned_depths.append(post_prune_depth)
        post_pruned_span = count_leaves(pruned_tree)
        post_pruned_spans.append(post_pruned_span)

        # separate features and labels for test data and evaluate tree
        X_test = test_split[:, :-1]
        Y_test = test_split[:, -1]
        evaluation_metrics = evaluation(X_test, Y_test, pruned_tree)

        # add the confusion matrix
        final_eval["confusion_matrix"] = (
            final_eval.get("confusion_matrix", 0)
            + evaluation_metrics["confusion_matrix"]
        )

        # update evaluation metrics
        # calculate sum of metrics for exach class
        for index_str in classes:
            for metric in ["precision", "recall"]:
                final_entry = final_eval.get(index_str, {})

                # safely get the value or default to 0 if not present
                final_eval[index_str] = final_entry
                final_eval[index_str][metric] = final_entry.get(
                    metric, 0
                ) + evaluation_metrics[index_str].get(metric, 0)

    # average metrics across all folds
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

    return final_eval, pre_pruned_depths, post_pruned_depths, pre_pruned_spans, post_pruned_spans


def make_data_folds(dataset, random_seed, k=10):
    """
    Evalute a simple tree on k-folds of the dataset

    Args:
        dataset: dataset to split
        random_seed: random seed
        k: number of folds
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

    # iterate through each fold
    for index, test_part in enumerate(test_parts):
        other_test_parts = [
            p for i, p in enumerate(test_parts) if i != index
        ]  # exclude current part by index

        # concatenate all other parts
        combined_data = np.vstack(other_test_parts)

        # separate features and labels for training data and train tree
        X_train = combined_data[:, :-1]  # all columns except the last
        Y_train = combined_data[:, -1]  # last column
        decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)

        # evaluate tree on test data
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
