import pandas as pd
# main algorithm depends on the specialized functions
import special_functions as sp


def type_of_features(df, values_threshold=5):
    """
    this api function return a tuple contain each feature type (continuous , categorical )
    :param values_threshold: the threshold of unique vals to consider the features categorical
    :param df: the data frame to determine its features types
    :return: sorted tuple contains each feature type
    """
    feature_types = []

    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= values_threshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return tuple(feature_types)


global FEATURES_NAMES


def train(df, ml_task, counter=0, min_samples=2, max_depth=5, ) -> str | dict:
    """
    This is the main functions that performs recursion algorithm to train and return a decision tree
     the tree in a shape of dict datastructure. don't use the counter parameter,
     the algorithm only uses it to handle the recursion in the most optimized way.
    :param ml_task: the task to perform classification or regression
    :param df: the pandas data frame to train the tree
    :param counter: (no for user )
    :param max_depth: the maximum depth of the tree
    :param min_samples: the min sample in a leaf
    :return: a tree data structure build on a dict
    """
    global FEATURES_NAMES  # a global var to store the feature names from first iteration

    # data preparations
    if counter == 0:  # when counter is 0 the function called for the first time with a pandas data frame
        # this is global vars for the features names and types
        FEATURES_NAMES = df.columns
        sp.FEATURE_TYPES = type_of_features(df)
        data = df.values

    # we convert it to numpy data array as all our algorithms runs on it
    #     counter not 0 means that recursion hasn't done yet
    else:
        data = df

    # base cases to stop recursion is when the splot data is pure,
    # or we reach the maximum depth or when collect the minimum number required as data samples
    if (sp.check_purity(data)) or (counter == max_depth) or (len(data) < min_samples):
        return sp.make_leaf(data, ml_task)

    # recursive part of the algorithm
    else:
        # counter will be used to get depth information
        counter += 1
        # dependencies on specialized functions
        split_feature_index, split_value = sp.fit_best_split(data, ml_task)  # first decide (fit best)
        # the most information given feature and value

        # to split the node on
        data_below, data_above = sp.split_data(data, split_feature_index, split_value)  # make the split

        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            return sp.make_leaf(data, ml_task)

        feature_name = FEATURES_NAMES[split_feature_index]
        feature_type = sp.FEATURE_TYPES[split_feature_index]
        # instantiate the question and the subtree(s)
        # for both continuous and categorical features
        if feature_type == "continuous":
            question = f"{feature_name} <= {split_value}"
        else:
            question = f"{feature_name} = {split_value}"

        sub_tree = {question: []}

        # find answers (in recursion way)
        yes_answer = train(data_below, ml_task, counter, min_samples, max_depth)

        no_answer = train(data_above, ml_task, counter, min_samples, max_depth)
        # respond to the caller with the answers

        if yes_answer == no_answer:  # to prevent the tree from creating such identical branches
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def predict_sample(sample, tree):
    """
    this API function classify one sample passed to it based on a given tree structure.
    :param sample: the sample to be classified
    :param tree: the tree structure to classify on
    :return: string a class name
    """
    # the question of the node is the key of the dict
    question = list(tree.keys())[0]

    # feature <= split_value
    # we split the attributes form the key
    feature_name, comparison_operator, value = question.split(" ")

    # ask question logically

    if comparison_operator == '<=':  # the feature is continuous
        if sample[feature_name] <= float(value):
            # the answer of yes leads to right branch
            answer = tree[question][0]
        else:
            # No leads to the left
            answer = tree[question][1]

    else:  # the feature is categorical
        if str(sample[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case to recurse in a subtree of return a class name
    if not isinstance(answer, dict):
        # answer is a class name
        return answer

    # recursive part , answer is another tree
    else:
        residual_tree = answer  # answer is a subtree
        return predict_sample(sample, residual_tree)


def predict(df, tree, ml_task):
    """
    This API function predict the classification of a pandas data frame based on a trained tree,
    Only used with classifier tree

    :param ml_task: the task to perform classification or regression
    :param df: the test pandas data frame
    :param tree: a trained tree to classify and predict on
    :return: a pandas data frame with new two columns,
    classification and prediction_correct
    """
    # add the classification col and apply the classify_sample function to each of its rows
    ret_df = df.copy()
    if ml_task != 'regression':
        ret_df["classification"] = df.apply(predict_sample, axis=1, args=(tree,))
        # add the prediction_correct col and set its values to booleans of the matching with the actual labels
        ret_df["prediction_correct"] = ret_df["classification"] == df["label"]
    elif ml_task == 'regression':
        ret_df["regression"] = df.apply(predict_sample, axis=1, args=(tree,))

    return ret_df


def calculate_accuracy(df):
    """
    This api function to calculate the accuracy of predicted classification from predict api.
    Only used with a classifier tree
    :param df: the pandas data frame returned from the dt_predict api
    :return: a float number represent the accuracy of the prediction
    """
    # accuracy is just the mean value of the binaries
    # in the prediction_correct col
    accuracy = df["prediction_correct"].mean()

    return accuracy


def calculate_r_squared(df, tree):
    labels = df.label
    mean = labels.mean()
    predictions = df.apply(predict_sample, args=(tree,), axis=1)

    ss_res = sum((labels - predictions) ** 2)
    ss_tot = sum((labels - mean) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return r_squared


def dt_tune_parameters(df, ml_task):
    if ml_task == 'regression':
        train_df, val_df = sp.train_test_split(df, test_size=0.3)
        grid_search = {"max_depth": [], "min_samples": [], "r_squared_train": [], "r_squared_val": []}
        for max_depth in range(1, 7):
            for min_samples in range(5, 20, 5):
                tree = train(train_df, ml_task="regression", max_depth=max_depth, min_samples=min_samples)

                r_squared_train = calculate_r_squared(train_df, tree)
                r_squared_val = calculate_r_squared(val_df, tree)

                grid_search["max_depth"].append(max_depth)
                grid_search["min_samples"].append(min_samples)
                grid_search["r_squared_train"].append(r_squared_train)
                grid_search["r_squared_val"].append(r_squared_val)

    else:  # task is classification
        grid_search = {"max_depth": [], "min_samples": [], "accuracy": []}
        for max_depth in range(1, 7):
            for min_samples in range(5, 20, 5):
                tree = train(df, ml_task="classification", max_depth=max_depth, min_samples=min_samples)

                accuracy = calculate_accuracy(predict(df, tree, 'classification'))

                grid_search["max_depth"].append(max_depth)
                grid_search["min_samples"].append(min_samples)
                grid_search["accuracy"].append(accuracy)

    grid_search = pd.DataFrame(grid_search)
    return grid_search.sort_values("r_squared_val", ascending=False)
