import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from collections.abc import Iterable

# Keras model optimizers
# NOTE: these are not usable without filling in the "learning_rate" arg; use with function interface
#   - RMSprop is gradient descent method based on rprop and like Adagrad with decay rate
#   - Adam is combo of Momentum and RMSprop, empirically great and go-to choice for deep learning
KERAS_OPTIMIZERS_DICT = {
    'rmsprop': tf.keras.optimizers.experimental.RMSprop,
    'adam': tf.keras.optimizers.Adam
}


def get_optimizer(optimizer_name, *optimizer_args, **optimizer_kwargs):
    """ Function interface for creating Keras optimizers, since they often need specific arguments
        NOTE: optimizers usually take learning_rate arg
    :param optimizer_name: string name by which we reference the optimizer
    :param optimizer_args: args to pass onto optimizer constructor
    :param optimizer_kwargs: kwargs to pass onto optimizer constructor
    :return: tf.keras.optimizers object
    """
    optimizer_name = optimizer_name.lower()
    return KERAS_OPTIMIZERS_DICT[optimizer_name](*optimizer_args, **optimizer_kwargs)


# Keras model metrics, mostly from:
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics
# NOTE: many of these are not usable without filling in the "thresholds" arg; use with function interface
KERAS_METRICS_DICT = {
    'mse': tf.keras.metrics.MeanSquaredError,
    'rmse': tf.keras.metrics.RootMeanSquaredError,
    'tp': tf.keras.metrics.TruePositives,
    'fp': tf.keras.metrics.FalsePositives,
    'tn': tf.keras.metrics.TrueNegatives,
    'fn': tf.keras.metrics.FalseNegatives,
    'accuracy': tf.keras.metrics.BinaryAccuracy,
    'precision': tf.keras.metrics.Precision,
    'recall': tf.keras.metrics.Recall,
    'auc': tf.keras.metrics.AUC
}


def get_metric(metric_name, *metric_args, **metric_kwargs):
    """ Function interface for creating Keras metrics, since they often need specific arguments
        NOTE: metrics usually take (classification) threshold arg
    :param metric_name: string name by which we reference the metric
    :param metric_args: args to pass onto metric constructor
    :param metric_kwargs: kwargs to pass onto metric constructor
    :return: tf.keras.metrics object
    """
    metric_name = metric_name.lower()
    if metric_name == 'prc':
        return tf.keras.metrics.AUC(name='prc', curve='PR')     # Precision-Recall curve
    else:
        return KERAS_METRICS_DICT[metric_name](*metric_args, **metric_kwargs)


def build_model(learning_rate, feature_layer=None, hidden_layers=None, binary=False, metrics=None):
    """ Create and compile a simple linear regression model
    :param learning_rate: chosen learning rate to pass into Keras optimizer
    :param feature_layer: tf.keras.layers.DenseFeatures()
    :param hidden_layers:
    :param binary:
    :param metrics:
    :return: compiled TensorFlow Keras model
    """
    # Most simple Keras models are sequential;
    # Sequential model contains one or more layers
    model = tf.keras.models.Sequential()

    # Describe topography of model
    #   - Simple linear regression: single node in a single layer
    #   - For multiple features, use feature layer (specifically constructed elsewhere)
    #   - For neural nets, add hidden layers (specifically constructed elsewhere)
    #   - For binary class classification, pipe through final sigmoid instead of single-node output layer
    if feature_layer is not None:
        # Add (bespoke constructed) layer containing feature columns
        model.add(feature_layer)
    if hidden_layers is not None:
        # Add (bespoke constructed) hidden layers for a deep neural net
        if isinstance(hidden_layers, Iterable):
            # Add each of many hidden layers
            for hidden_layer in hidden_layers:
                model.add(hidden_layer)
        else:
            # Input is likely a single layer object - add it
            model.add(hidden_layers)
    # Finish with output layer
    if binary:
        # Funnel regression value through sigmoid function for binary classification
        sigmoid_layer = tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid, name='Output')
        model.add(sigmoid_layer)
    else:
        # Add one linear layer to yield simple linear regressor
        one_node_layer = tf.keras.layers.Dense(units=1, input_shape=(1,), name='Output')
        model.add(one_node_layer)

    # Compile model topography into efficient TF code
    # Configure training to minimize model's loss - whether its mean squared error or other
    if binary:
        loss = tf.keras.losses.BinaryCrossentropy()     # Binary cross-entropy loss
    else:
        loss = 'mean_squared_error'     # String shortcut for MSE loss
    if metrics is None:
        metrics = [get_metric('rmse', name='rmse')]
    model.compile(optimizer=get_optimizer('Adam', learning_rate=learning_rate),
                  loss=loss,
                  metrics=metrics)

    return model


def train_model(model, dataset_df, label_name, n_epochs, batch_size=None, validation_split=0.2, shuffle=True):
    """ Train model by feeding it data
    :param model: compiled Tensorflow Keras model
    :param dataset_df:
    :param label_name: string name of column in dataset_df to use as label;
                       don't need feature names because we pass rest of dataset_df to model, and
                       we assume model has feature columns which already define their names
    :param n_epochs:
    :param batch_size:
    :param validation_split:
    :param shuffle: set True to shuffle training data before each epoch
    :return:
    """
    # Feed feature values and label values to model;
    # Model will train for specified number of epochs, gradually learning how
    # feature values relate to label values
    # NOTE: model.fit() can take dict of arrays and filter away columns based on model feature layer
    features_df = dataset_df.copy()     # Will modify copy to become just features
    label_ser = features_df.pop(label_name)
    history = model.fit(x=dict(features_df),
                        y=label_ser,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        shuffle=shuffle)

    # Gather the trained model's weight and bias
    # NOTE: .item() can extract number from np.array wrapper
    trained_weight = model.get_weights()[0]     # Can be multiple weights with multiple features
    trained_bias = model.get_weights()[1]   # Can be multiple biases with hidden layers (apparently)

    # Gather snapshot of each epoch
    history_df = pd.DataFrame(history.history)
    epochs_list = history.epoch  # list of epochs, stored separately from rest of history

    return epochs_list, history_df, trained_weight, trained_bias


def plot_the_model(trained_weight, trained_bias, df, feature, label):
    """ Plot the trained model against the training feature and label
    :param trained_weight:
    :param trained_bias:
    :param df:
    :param feature:
    :param label:
    :return:
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Label axes
    ax.set_xlabel(f"Feature: {feature}")
    ax.set_ylabel(f"Label: {label}")

    # Create scatter plot of feature values vs. label values
    random_examples = df.sample(n=200)  # Pick 200 random points from dataset
    ax.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model; line starts at (x0, y0) and ends at (x1, y1)
    x0, y0 = 0, trained_bias
    x1 = random_examples[feature].max()     # Biggest feature value; assumes everything linear
    y1 = trained_bias + trained_weight*x1
    ax.plot([x0, x1], [y0, y1], color='r')

    # Render figure
    plt.show()


def plot_the_loss_curve(epochs_list, training_loss, validation_loss):
    """ Plot the loss curve, which shows loss vs. epoch
    :param epochs_list:
    :param training_loss:
    :param validation_loss:
    :return:
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Label axes
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Root Mean Squared Error")

    # Create line of x-axis number of epoch vs. y-axis loss; omit first epoch which is distractingly bad
    ax.plot(epochs_list[1:], training_loss[1:], label="Training Loss")
    ax.plot(epochs_list[1:], validation_loss[1:], label="Validation Loss")
    ax.legend()
    merged_loss_values = pd.concat([training_loss, validation_loss])
    highest_loss, lowest_loss = max(merged_loss_values), min(merged_loss_values)
    delta = highest_loss - lowest_loss
    ax.set_ylim([lowest_loss-delta*0.05, highest_loss+delta*0.05])

    # Render figure
    plt.show()


def plot_metrics_curve(epochs_list, history_df, metrics=None):
    """ Plot curve of one or more classification metrics vs. epoch
    :param epochs_list:
    :param history_df:
    :param metrics:
    :return:
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    # Label axes
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    # Create line of x-axis epochs vs. y-axis metric for each metric
    if metrics is None:
        metrics = [get_metric('rmse', name='rmse')]
    metric_names = [metric.name for metric in metrics]  # Extract string names
    for metric_name in metric_names:
        metric_hist = history_df[metric_name]
        ax.plot(epochs_list[1:], metric_hist[1:], label=metric_name)
    ax.legend()
    # Render figure
    plt.show()


def predict_random_batch(model, dataset_df, label_name, n_samples):
    """ TODO: not in readable format with multiple features
    Predict house values based on a feature."""
    selected_features_df = dataset_df.sample(n_samples)     # Will modify copy to become just features batch
    selected_label_ser = selected_features_df.pop(label_name)
    predicted_values = model.predict_on_batch(x=dict(selected_features_df))
    print("feature value\tlabel value\tpredicted value\n"
          "-------------------------------------------\n")
    for i in range(n_samples):
        print(f"{selected_features_df.iloc[i]}\t"
              f"{selected_label_ser.iloc[i]:.1f}\t"
              f"{predicted_values[i][0]:.1f}\n")


def normalize_columns(dataset, ddof=1):
    """ Convert each column of dataset to Z-score based on column mean and std
    :param dataset: pandas DataFrame or Series with column(s)
    :param ddof: Delta Degrees of Freedom, divisor in std is N-ddof;
                 pandas std() defaults to 1 (sample std, not population)
    :return: pd.DataFrame with same dimensions as input dataset_df
    """
    dataset_mean = dataset.mean()   # Mean of each column (collapse rows)
    dataset_std = dataset.std(ddof=ddof)
    dataset_norm = (dataset - dataset_mean) / dataset_std  # Z-score
    return dataset_norm


###############################################################################

if __name__ == '__main__':
    # Import California Housing Dataset
    train_df = pd.read_csv(
        filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    test_df = pd.read_csv(
        filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
    train_df = train_df.sample(frac=1)  # Shuffle to remove validation set bias, but don't reset index
    train_df.head()  # PRINT first rows
    train_df.describe()  # PRINT stats
    train_df.corr()  # PRINT correlation matrix

    # Normalize values to Z-score (except lat/long coordinates, preserve those for one-hot)
    train_df_norm = normalize_columns(train_df)
    # train_df_norm['latitude'], train_df_norm['longitude'] = train_df['latitude'], train_df['longitude']
    train_df_norm.head()    # PRINT for sanity check of normalized numbers
    test_df_norm = normalize_columns(test_df)
    # test_df_norm['latitude'], test_df_norm['longitude'] = test_df['latitude'], test_df['longitude']

    # Create binary label for "is median house value higher than 75th percentile?"
    threshold_Z = train_df_norm['median_house_value'].quantile(0.75)    # $265,000 in non-Z-score
    train_df_norm['median_house_value_is_high'] = train_df_norm['median_house_value'].gt(threshold_Z).astype(float)
    test_df_norm['median_house_value_is_high'] = test_df_norm['median_house_value'].gt(threshold_Z).astype(float)

    # Choose label
    my_label_name = 'median_house_value'

    # Specify our chosen feature(s)
    feature_columns = []    # Initialize list of feature columns
    # 1) Feature cross for latitude X longitude
    resolution_in_Z = 0.3   # 3/10 of a standard deviation in degrees - yes it's weird
    # Create bucketized feature column for latitude
    latitude_num_col = tf.feature_column.numeric_column('latitude')    # Numerical feature
    latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])),
                                         int(max(train_df_norm['latitude'])),
                                         resolution_in_Z))
    latitude_bucket_col = tf.feature_column.bucketized_column(latitude_num_col, latitude_boundaries)    # Bucket
    # Create bucketized feature column for longitude
    longitude_num_col = tf.feature_column.numeric_column('longitude')  # Numerical feature
    longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])),
                                          int(max(train_df_norm['longitude'])),
                                          resolution_in_Z))
    longitude_bucket_col = tf.feature_column.bucketized_column(longitude_num_col, longitude_boundaries)     # Bucket
    # Create a feature cross (one-hot indicator, not just multiplication) of latitude and longitude
    latitude_x_longitude_col = tf.feature_column.crossed_column([latitude_bucket_col, longitude_bucket_col],
                                                                hash_bucket_size=100)
    latitude_x_longitude_indicator_col = tf.feature_column.indicator_column(latitude_x_longitude_col)   # One-hot
    feature_columns.append(latitude_x_longitude_indicator_col)
    # 2) Numerical feature column for median_income
    median_income_num_col = tf.feature_column.numeric_column('median_income')
    feature_columns.append(median_income_num_col)
    # 3) Numerical feature column for population
    population_num_col = tf.feature_column.numeric_column('population')
    feature_columns.append(population_num_col)
    # Convert list of feature columns into a layer for model
    my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # Specify our hidden layers
    my_hidden_layers = []
    # First hidden layer with 20 nodes
    hidden_layer_1 = tf.keras.layers.Dense(units=20, activation='relu', name='Hidden1',
                                           kernel_regularizer=tf.keras.regularizers.l2(l=0.04))
    my_hidden_layers.append(hidden_layer_1)
    # # Add dropout regularization as a separate layer
    # dropout_regularization_layer = tf.keras.layers.Dropout(rate=0.25)
    # my_hidden_layers.append(dropout_regularization_layer)
    # Second hidden layer with 12 nodes
    hidden_layer_2 = tf.keras.layers.Dense(units=12, activation='relu', name='Hidden2',
                                           kernel_regularizer=tf.keras.regularizers.l2(l=0.04))
    my_hidden_layers.append(hidden_layer_2)

    ############################################################################
    # Following section will be re-run empirically and tuned

    # Set proportion of training set to split off as validation set
    my_validation_split = 0.2
    # Tune the hyperparameters
    my_learning_rate = 0.007
    my_n_epochs = 140
    my_batch_size = 1000
    # my_classification_threshold = 0.52

    # # Establish the binary class classifcation metrics the model will measure
    # my_metrics = [
    #     tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=my_classification_threshold),
    #     tf.keras.metrics.Precision(name='precision', thresholds=my_classification_threshold),
    #     tf.keras.metrics.Recall(name="recall", thresholds=my_classification_threshold),
    #     tf.keras.metrics.AUC(name='auc', num_thresholds=100)
    # ]

    # Use our modeling functions
    my_model = build_model(my_learning_rate, my_feature_layer, my_hidden_layers,
                           binary=False, metrics=[get_metric('mse', name='mse')])
    epochs, hist_df, weight, bias = \
        train_model(my_model, train_df_norm, my_label_name, my_n_epochs, my_batch_size, my_validation_split)
    print(f"\nYour feature: DISABLED FOR MULTIPLE FEATURES"
          f"\nYour label: {my_label_name}"
          f"\nThe learned weight(s) for your model: {weight}"
          f"\nThe learned bias for your model: {bias}\n")

    # Use our visualization functions
    # plot_the_model(weight, bias, train_df, my_feature, my_label)
    # plot_the_loss_curve(epochs, rmse_train, rmse_val)
    plot_metrics_curve(epochs, hist_df, metrics=[get_metric('mse', name='mse')])

    # Sanity check a random batch of predictions
    # predict_random_batch(my_model, train_df_norm, my_label_name, 15)

    # Ultimate judgment: use test set to evaluate the model
    test_features_df = test_df_norm.copy()
    test_label_ser = test_features_df.pop(my_label_name)
    results = my_model.evaluate(dict(test_features_df), test_label_ser, batch_size=my_batch_size)
