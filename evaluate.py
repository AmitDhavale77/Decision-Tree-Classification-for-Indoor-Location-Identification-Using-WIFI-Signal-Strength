import numpy as np
from main import decision_tree_learning

# Do evaluation stuff


# evaluation_example = {
#     '1': {
#         'accuracy': None,
#         'precision': None,
#         'recall': None,
#         'f1': None,
#     },
#     # etc... for 2, 3, 4
#     'confusion_matrix': None # 4x4 matrix
#     'overall': {
#         # same as above

#     }
# }


def compute_accuracy(y_gold, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        accuracy (float): accuracy value between 0 and 1
    """

    assert len(y_gold) == len(y_prediction)

    if len(y_gold) == 0:
        return 0

    return np.sum(y_gold == y_prediction) / len(y_gold)


def predictions(tree, test_x):
    """ 
    Predict output values from a tree using a test set

    Args:
        tree(dict): trained decision tree dictionary 
        test_x(np.array): array of x features to get classification predictions from the tree

    Returns:
        np.array: array of size len(test_x) of output predictions from tree

    """

    num_rows, _ = test_x.shape
    predictions = np.zeros((num_rows,))

    for row in range(num_rows):
        test_row = test_x[row, :]
        predictions[row] = row_predict(tree, test_row)

    return predictions


def row_predict(tree, test_row):
    """ 
    Make a prediction for a single row on an input dataset using a trained tree

    Args: 
        tree(dict): trained decision tree dictionary
        test_row(np.array): single row of an np.array to make a decision from
    """

    if tree['feature'] is None:
        return tree['value']

    cur_feature = tree['feature']
    cur_value = tree['value']
    go_right = test_row[cur_feature] > cur_value
    if go_right:
        return row_predict(tree['right'], test_row)
    else:
        return row_predict(tree['left'], test_row)


def get_classification_evaluation():
    """ return dictionary in format: 
            {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
            }
    """
    pass


def get_confusion_matrix(y_gold, y_prediction):
    """
    create the confusion matrix from the true and predicted values

    Args: 
        y_gold: TODO
        y_prediction 

    Returns:
        np.array of size n_classifications x n_classifications
    """

    # get number of unique classifications (assuming they all appear in the data)
    n_classifications = 4 #TODO

    # create empty matrix
    confusion_matrix = np.zeros((n_classifications, n_classifications))

    return confusion_matrix


def evaluation(test_db, trained_tree):
    """
    @Lauren
    Confusion Matrix (4x4 matrix)
    Then for each label:
        Accuracy
        Precision/Recall
        F1 score
    Then do macro-averaging to get overall
    Should we do micro-averaging??? TBD

    Return a dictionary with all of these metrics
    """
    # predict output 
    x_test = test_db[:,:-1]
    y_gold = test_db[:, -1]
    y_prediction = predictions(trained_tree, x_test)

    # evaluate results
    confusion_matrix = get_confusion_matrix(y_gold, y_prediction)

    evaluation = {"confusion_matrix":confusion_matrix}

    classes = [str(classification) for classification in test_db[:,-1].unique()]
    for classification in classes: 
        evaluation[classification] = get_classification_evaluation()

    # macroaverage overall metrics 
    evaluation['overall'] = {}

    return evaluation



def advanced_k_fold(dataset, k=10):
    # @Amit
    # implement option 2 from slide 30 for cross-validation 
    # this will be used for parameter tuning the pruning part

    # this will call make_data_folds

    # Return evaluation dictionary in the same format as above one but
    # averaged across the k folds

    # this will use prune_tree to test on different trees
    return


def make_data_folds(dataset, random_seed, k=10):
    # @Ola
    # Look at existing version of train_test_split but will need to make folds deterministically

    # Shuffle data
    # Then pick first two rows for test, rest is training+validation
    # Evaluate that data
    # then pick next two rows, cont...
    # As you go through, keep track of averaged evaluation metrics (all of them)

    # Return evaluation dictionary in the same format as above one but
    # averaged across the k folds
    # Create random generator

    #what is the split is not equal??
    random_generator = np.random.default_rng(random_seed)

    # Create array of shuffled indices
    shuffled_indices = random_generator.permutation(len(dataset))

    data_shuffled = dataset[shuffled_indices]

    parts = np.split(data_shuffled, 10, axis=0)

    final_eval = {}

    evaluation_example = {
     '1': {
        'precision': None,
        'recall': None,
        'f1': None,
     },
     '2': {
        'precision': None,
        'recall': None,
        'f1': None,
     },
     '3': {
        'precision': None,
        'recall': None,
        'f1': None,
     },
     '4': {
        'precision': None,
        'recall': None,
        'f1': None,
     },
     'confusion_matrix': None, # 4x4 matrix
     'accuracy': None,
     'overall': {
        'precision': None,
        'recall': None,
        'f1': None,
     }
 }

    for index, part in enumerate(parts):
        other_parts = [p for i, p in enumerate(parts) if i != index]  # Exclude current part by index

        # Concatenate all other parts
        combined_data = np.vstack(other_parts)  # Stack all other parts vertically

        # Separate features (X) and labels (Y)
        X_train = combined_data[:, :-1]  # All columns except the last
        Y_train = combined_data[:, -1]   # Last column

        decision_tree, tree_depth = decision_tree_learning(X_train, Y_train)

        evaluation_metrics = evaluation(part, decision_tree)

        # add the confusion matrix
        final_eval['confusion_matrix'] = final_eval.get('confusion_matrix', 0) + evaluation_metrics['confusion_matrix']

        # add the accuracy
        final_eval['accuracy'] = final_eval.get('accuracy', 0) + evaluation_metrics['accuracy']

        # calculate overall metrics

        """
        final_entry = final_eval.get('overall', {})
        final_eval['overall'] =  final_entry 
        final_eval['overall']['precision'] = final_entry.get('precision', 0) + evaluation_metrics['overall']['precision']
        final_eval['overall']['recall'] = final_entry.get('recall', 0) + evaluation_metrics['overall']['recall']
        final_eval['overall']['f1'] = final_entry.get('f1', 0) + evaluation_metrics['overall']['f1']
        """

        # calculate sum of metrics for exach class
        for index in range(1,4):
            index_str = str(index)
            for word in ['precision', 'recall', 'f1']:
                final_entry = final_eval.get(index_str, {})
                
                # Safely get the value or default to 0 if not present
                final_eval[index_str] = final_entry  # Ensure it exists in final_eval
                final_eval[index_str][word] = final_entry.get(word, 0) + evaluation_metrics[index_str].get(word, 0)
    
    # calculate averages across all folds
    for index in range(1,4):
        index_str = str(index)
        for word in ['precision', 'recall', 'f1']:
            final_eval[index_str][word] /= 10

    final_eval['accuracy'] /= 10

    # calculate macro-averaged overall metrics
    final_eval['overall']['precision'] = (final_eval['1']['precision']+final_eval['2']['precision']+final_eval['3']['precision']+final_eval['4']['precision'])/4
    final_eval['overall']['recall'] = (final_eval['1']['recall']+final_eval['2']['recall']+final_eval['3']['recall']+final_eval['4']['recall'])/4
    final_eval['overall']['f1'] = (final_eval['1']['f1']+final_eval['2']['f1']+final_eval['3']['f1']+final_eval['4']['f1'])/4

    return final_eval 

matrix = np.array([
    [95, 20, 74, 83, 18, 96, 82, 77, 89, 43],
    [17, 24,  4, 93, 48, 18, 22, 15, 15, 19],
    [63,  6, 73, 89, 15, 20, 92, 43, 10, 83],
    [59, 87, 59, 53, 72,  7, 36, 45, 37, 49],
    [25, 71, 48,  1, 19, 41, 76, 58, 89, 48],
    [39,  6, 46, 44, 79, 43, 81, 56, 20, 52],
    [94, 42,  3, 88, 41, 29, 80, 27, 81, 85],
    [33, 19,  9, 17, 53, 15, 34, 94, 96, 56],
    [51, 87, 45, 90, 38, 82,  9, 52, 43, 37],
    [91, 16, 78,  5, 41, 97, 81, 64, 46, 13],
])

print(make_data_folds(matrix, random_seed=0))

