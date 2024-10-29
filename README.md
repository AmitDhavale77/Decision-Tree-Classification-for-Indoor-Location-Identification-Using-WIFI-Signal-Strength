# Decision Tree Coursework

This code will produce all of the outputs required to answer the questions in 
the coursework, and optionally generating a visulisation of a decision tree. 

The Python version and library requirements are in requirements.txt

## Project Structure

#### `main.py`

This script is organizes all the subsequent scripts into a full project pipeline.

#### `data_prep.py`

This script contains functions to load the data from file, split and sort it.

#### `decision_tree.py`

This script contains the function to train the decision tree and related helper functions
such as calculating entropy and information gain, finding the ideal split points and making predictions.

#### `visualize.py`

This script contains functions to plot a visualization of the decision tree.

#### `prune.py`

This script contains functions to prune the tree as well as helper functions to find candidate nodes and replace them.

#### `evaluate.py`

This script contains functions to determine the confusion matrix, accuracy and other evaluation metrics.

#### `cross_validation.py`

This script contains functions to do nested k folds cross-validation.

## Instructions

Running the script to train a decision tree and evaluate it:

```bash
python main.py
```

Configuration Options:

- Input Data File: change main.py line 76 (default: clean_data.txt)
- \# of Data Folds (k): change main.py line 77 (default: 10)

### Run Modes

#### run_demo()

This will train a decision tree and report the evaluation metris of a demo unpruned and pruned tree.

#### run_question_3()

This will report the average evaluation metrics for a 10 fold cross validation on unpruned trees.

#### run_question_4()

This will report the average evaluation metrics for a nested 10 fold cross validation on pruned trees.
