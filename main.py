import numpy as np

from evaluate import compute_average_evaluation, simple_compute_accuracy
from visualize import visualize_tree, tree_to_json, json_to_tree
from data_prep import load_data, train_test_split
from decision_tree import decision_tree_learning, get_tree_depth, count_leaves
from cross_validation import nested_k_folds


if __name__ == "__main__":
    # import and split data
    clean_filename = "wifi_db/clean_dataset.txt"
    noisy_filename = "wifi_db/noisy_dataset.txt"
    X, Y = load_data(clean_filename)

    ####################### Question 3
    test_proportion = 0.6
    random_seed = 42

    X_train, Y_train, X_test, Y_test = train_test_split(
        X, Y, test_proportion, random_seed
    )

    decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)
    og_depth = get_tree_depth(decision_tree)
    og_span = count_leaves(decision_tree)
    og_accuracy = simple_compute_accuracy(decision_tree, X_test, Y_test)
    print(f"Original Depth: {og_depth} & Span: {og_span}")
    print(f"Original Accuracy: {og_accuracy}")

    ####################### Visulize Tree
    visualize_tree(decision_tree, og_depth)

    ####################### Question 4
    # nested k-folds
    eval_list = nested_k_folds(X, Y, k=10, random_seed=60012)

    # report the average results of the k sets of trees
    accuracy, confusion_matrix, class_dict = compute_average_evaluation(eval_list)
    print("accuracy", accuracy)
    print("confusion_matrix", confusion_matrix)
    print("class_dict", class_dict)


################################################################################################


# eval_dict = evaluation(X_test, Y_test, decision_tree)
# tree_to_json(decision_tree, "tree.json")
# decision_tree = json_to_tree("noisy_tree.json")
# visualize_tree(decision_tree, 11)
# print(f"Accuracy on test set: {accuracy*100}%")
# print(eval_dict)

# pruned_tree = prune_tree(decision_tree, X_test, Y_test)
# new_depth = get_tree_depth(pruned_tree)
# new_span = count_leaves(pruned_tree)
# new_accuracy = simple_compute_accuracy(pruned_tree, X_test, Y_test)

# print(f"Pruned Depth: {new_depth} & Span: {new_span}")
# print(f"New Accuracy: {new_accuracy}")

# # Nested K-folds starts from here
# accuracy, t_matrix, class_dict = compute_overall_average_nested_kfolds(eval_list)

# print("accuracy", accuracy)
# print("t_matrix", t_matrix)
# print("class_dict", class_dict)

# data = np.loadtxt(noisy_filename)
# print(make_data_folds(data, 42, k=10))

# visualize_tree(pruned_tree, new_depth)
