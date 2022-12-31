import numpy as np
import random

# Specialized Functions module
global FEATURE_TYPES


# for data preparation
def train_test_split(df, test_size) -> tuple:
    """
    this function splits the data frame into to sub-frames, first for training the model second to test.
    it's optionally to set the random seed to constant value for constant splits for each call.
    :param df: pandas data frame to split.
    :param test_size: the portion or the number of samples to pick fot the test frame
    :return: tuple contains two pandas data frame (train_data , test_data)
    """
    # checks if the test_size is a portion or amount
    if isinstance(test_size, float):
        # if portion calculate the amount
        test_size = round(test_size * len(df))
    # generate a list of indices in the data frame
    indices = df.index.tolist()

    # pick the indices of the amount required randomly
    test_indices = random.sample(population=indices, k=test_size)

    # locate the test indices to a new dataframe (test dataframe)
    test_df = df.loc[test_indices]
    # drop the test indices to get the train data frame
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data) -> bool:
    """
    This function checks for purity in the data before any attempt to split
    :param data: data array to check its purity
    :return: true if pure , false if not
    """
    # get an array of the label column
    label_column = data[:, -1]
    # get the unique classes from the label column
    unique_classes = np.unique(label_column)
    # the data is pure if there is one class
    return len(unique_classes) == 1


def make_leaf(data, ml_task) -> str | float:
    """
    This function return the most occurred class in the data array passed to it
    if ml_task is classification and the mean of the label vals if the ml_task is regression.

    :param data: the array of data to make a leaf node
    :param ml_task: string 'classification' of 'regression'
    :return: float if regression , str if classification
    """
    # generate array of the label column
    label_column = data[:, -1]
    if ml_task == "regression":
        leaf = np.mean(label_column)
    else:  # classification
        # get the unique classes from the label column as well as number of occurrence for each class
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        # return a string of the name for the most occurred one
        index_of_max = counts_unique_classes.argmax()
        leaf = unique_classes[index_of_max]

    return leaf


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


def set_feature_types(types):
    global FEATURE_TYPES
    FEATURE_TYPES = types


def get_potential_splits(data) -> dict:
    """
    This function return a dic contains the indices of the continues features in the data as keys
    and the values for potentially split on for each feature.
    """
    # create the dic to be returned
    potential_splits = {}
    # get the dims of the frame
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        # generate an array of contained the values of this col
        values = data[:, column_index]
        # extract the uniques
        unique_values = np.unique(values)
        # for each index create a list as key
        potential_splits[column_index] = list(unique_values)

    return potential_splits


def split_data(data, split_column, split_value):
    """ this function split the data frame into two sub-frames based on a split value in the split col """
    feature_type = FEATURE_TYPES[split_column]
    # get an array of the values in the split col
    split_column_values = data[:, split_column]

    # if the feature is continuous
    if feature_type == 'continuous':
        # for each value below than the threshold select its row in the data_below frame
        data_below = data[split_column_values <= split_value]
        # for each value above than the threshold select its row in the data_above frame
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def entropy(data):
    """
    this function calculate and return the entropy of the given data frame.
    :param data: data to calculate entropy for
    :return: float number represent the entropy of the data
    """

    # we get an array contains all values in label col (classes)
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)  # counts the occurrence of each class
    # get all classes probabilities
    probabilities = counts / counts.sum()
    # calculate entropy the weighted sum of the -log all probabilities
    c_entropy = sum(probabilities * -np.log2(probabilities))

    return c_entropy


def mse(data):
    """
    this function returns the mean squared error of the data given to it.
    :param data: data to calculate mse for.
    :return: float number represents the mse value.
    """
    # we get an array contains all values in label col
    actual_values = data[:, -1]
    # checks for empty array of data , then mse is zero
    if len(actual_values) == 0:  # empty data
        calc_mse = 0

    else:  # data frame is not empty
        # mse = (1/n) * sum(actual - predict)^2
        # So, we first get the predicted value as it is the mean of all values.
        prediction = np.mean(actual_values)
        calc_mse = np.mean((actual_values - prediction) ** 2)

    return calc_mse


def overall_metric(data_below, data_above, metric_method):
    """ this function calculate the overall gain of splot data ,
    E(data ) - ( w1.E(left_branch) + w2.E(right_branch) )  """
    # calculate the wights of the two branches
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    # calculate overall entropy and return it
    overall = (p_data_below * metric_method(data_below)
               + p_data_above * metric_method(data_above))

    return overall


def info_gain(data, data_below, data_above, metric_method):
    """ This functions returns the information gain we get from a specified split """
    # information gain is simply the entropy of whole data - overall_entropy
    return metric_method(data) - overall_metric(data_below, data_above, metric_method)


def fit_best_split(data, ml_task: str):
    """This function returns the index of the  best fit feature and the best fit value to split the data on  """
    # get all potential_splits that features could split on
    potential_splits = get_potential_splits(data)
    # set the max info gain to zero and iterate over all feature with all split values
    # searching for that feature and split value
    first_iteration = True
    lowest_metric = None
    best_split_feature, best_split_value = None, None
    for feature in potential_splits:
        for split_value in potential_splits[feature]:
            # we split for the current feature and split value
            data_below, data_above = split_data(data, feature, split_value)
            # if the task is regression calculate the gain based of the mse
            if ml_task == 'regression':
                current_metric = overall_metric(data_below, data_above, mse)
            #  if the task is classification calculate gain based on entropy
            else:  # classification
                current_metric = overall_metric(data_below, data_above, entropy)
            # if the current gain higher than the max update our vars
            if first_iteration or (current_metric <= lowest_metric):
                first_iteration = False
                lowest_metric = current_metric
                best_split_feature = feature
                best_split_value = split_value

    return best_split_feature, best_split_value
