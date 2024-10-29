import numpy as np

from evaluate import (
    compute_average_evaluation,
    evaluation,
    report_evaluation,
)
from visualize import visualize_tree
from data_prep import load_data, train_test_split
from decision_tree import decision_tree_learning, get_tree_depth, count_leaves
from cross_validation import nested_k_folds, make_data_folds
from prune import prune_tree


def run_demo(filename, test_proportion=0.6, random_seed=42):
    """Run a demo on a single tree before and after pruning"""
    print("\n\n\n\n         Demo of a single tree")
    X, Y = load_data(filename)

    # split data, train and evaluate tree model
    X_train, Y_train, X_test, Y_test = train_test_split(
        X, Y, test_proportion, random_seed
    )
    decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)
    eval_dict = evaluation(X_test, Y_test, decision_tree)

    # report tree size, evaluation metrics and visualize tree
    og_depth = get_tree_depth(decision_tree)
    og_span = count_leaves(decision_tree)
    print(f"Example of a tree without pruning")
    print(f"Original Depth: {og_depth} & Span: {og_span}")
    report_evaluation(eval_dict)
    visualize_tree(decision_tree, og_depth)

    # prune tree and evaluate
    pruned_tree = prune_tree(decision_tree, X_test, Y_test)
    new_depth = get_tree_depth(pruned_tree)
    new_span = count_leaves(pruned_tree)
    eval_dict = evaluation(X_test, Y_test, pruned_tree)

    # report pruned tree size, evaluation metrics
    print("Example of a single tree with pruning:")
    print(f"Pruned Depth: {new_depth} & Span: {new_span}")
    report_evaluation(eval_dict)
    visualize_tree(pruned_tree, new_depth)


def run_question_3(filename, test_proportion=0.6, random_seed=42, k=10):
    """Run question 3 of the coursework, cross validation and pruning"""
    print("\n\n\n\n         Question 3")

    # run prune cross validation
    print("\nExample of k-folds to evaluate trees:")
    data = np.loadtxt(filename)
    eval_dict = make_data_folds(data, 42, k=k)
    report_evaluation(eval_dict)


def run_question_4(filename, k=10, random_seed=60012):
    """Run question 4 of the coursework, nested cross validation"""
    print("\n\n\n\n         Question 4")

    # load data and run nested k-folds
    X, Y = load_data(filename)
    eval_list = nested_k_folds(X, Y, k, random_seed)
    classes = [str(float(classification)) for classification in np.unique(Y)]

    # report the average results of the k sets of trees
    eval_dict = compute_average_evaluation(eval_list, classes)
    report_evaluation(eval_dict)


if __name__ == "__main__":
    ####################### Set filenames and k
    clean_filename = "wifi_db/clean_dataset.txt"
    noisy_filename = "wifi_db/noisy_dataset.txt"

    demo_file = clean_filename
    demo_k = 10

    ####################### demo a single tree (visulise)
    # uncomment to run visulisation of a single tree pre and post pruning
    # run_demo(filename=demo_file)

    ####################### Question 3
    run_question_3(filename=demo_file, k=demo_k)

    # ####################### Question 4
    run_question_4(filename=demo_file, k=demo_k)
