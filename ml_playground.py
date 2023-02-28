import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Keras model metrics, mostly from:
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics
# NOTE: many of these are not usable without filling in the "thresholds" arg,
#       so this dictionary is mainly for copy-pasting reference at the moment
KERAS_METRICS_DICT = {
    'root_mean_squared_error': tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
    'tp': tf.keras.metrics.TruePositives(name='tp'),
    'fp': tf.keras.metrics.FalsePositives(name='fp'),
    'tn': tf.keras.metrics.TrueNegatives(name='tn'),
    'fn': tf.keras.metrics.FalseNegatives(name='fn'),
    'accuracy': tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    'precision': tf.keras.metrics.Precision(name='precision'),
    'recall': tf.keras.metrics.Recall(name='recall'),
    'auc': tf.keras.metrics.AUC(name='auc'),
    'prc': tf.keras.metrics.AUC(name='prc', curve='PR')
}


def build_model(learning_rate, feature_layer=None, binary=False, metrics=None):
    """ Create and compile a simple linear regression model
    :param learning_rate: chosen learning rate to pass into Keras optimizer
    :param feature_layer: tf.keras.layers.DenseFeatures()
    :param binary:
    :param metrics:
    :return: compiled TensorFlow Keras model
    """
    # Most simple Keras models are sequential;
    # Sequential model contains one or more layers
    model = tf.keras.models.Sequential()

    # Describe topography of model
    #   - Simple linear regression model: single node in a single layer
    #   - For multiple features, use feature layer
    if feature_layer is not None:
        model.add(feature_layer)    # Add layer containing feature columns
    if binary:
        # Funnel regression value through sigmoid function for binary classification
        sigmoid_layer = tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid)
        model.add(sigmoid_layer)
    else:
        # Add one linear layer to yield simple linear regressor
        one_node_layer = tf.keras.layers.Dense(units=1, input_shape=(1,))
        model.add(one_node_layer)

    # Compile model topography into efficient TF code
    # Configure training to minimize model's mean squared error
    #   - RMSprop is gradient descent method based on rprop and similar to Adagrad
    rmsprop_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)
    mse_loss = 'mean_squared_error'     # String shortcut
    binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()
    if binary:
        loss = binary_crossentropy_loss
    else:
        loss = mse_loss
    model.compile(optimizer=rmsprop_optimizer,
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
    trained_bias = model.get_weights()[1].item()

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


def plot_metrics_curve(epochs_list, history_df, metrics):
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
    train_df.head()  # Print first rows
    train_df.describe()  # Print stats
    train_df.corr()  # Print correlation matrix

    # Normalize values to Z-score (except lat/long coordinates, preserve those for one-hot)
    train_df_norm = normalize_columns(train_df)
    train_df_norm['latitude'], train_df_norm['longitude'] = train_df['latitude'], train_df['longitude']
    train_df_norm.head()    # Print normalized to sanity check
    test_df_norm = normalize_columns(test_df)
    test_df_norm['latitude'], test_df_norm['longitude'] = test_df['latitude'], test_df['longitude']

    # Create binary label for "is median house value higher than 75th percentile?"
    threshold_Z = train_df_norm['median_house_value'].quantile(0.75)    # 265,000 in non-Z-score
    train_df_norm['median_house_value_is_high'] = train_df_norm['median_house_value'].gt(threshold_Z).astype(float)
    test_df_norm['median_house_value_is_high'] = test_df_norm['median_house_value'].gt(threshold_Z).astype(float)

    # Specify our chosen feature(s) and label
    feature_columns = []    # Initialize list of feature columns
    # 1) Numerical feature column for median_income
    median_income_num_col = tf.feature_column.numeric_column('median_income')
    feature_columns.append(median_income_num_col)
    # 2) Numerical feature column for total_rooms
    tr_num_col = tf.feature_column.numeric_column('total_rooms')
    feature_columns.append(tr_num_col)
    # # 3) Feature cross for latitude X longitude
    # resolution_in_degrees = 0.4
    # # Create bucketized feature column for latitude
    # latitude_num_col = tf.feature_column.numeric_column('latitude')    # Numerical feature
    # latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
    #                                      int(max(train_df['latitude'])),
    #                                      resolution_in_degrees))
    # latitude_bucket_col = tf.feature_column.bucketized_column(latitude_num_col, latitude_boundaries)    # Bucket
    # # Create bucketized feature column for longitude
    # longitude_num_col = tf.feature_column.numeric_column('longitude')  # Numerical feature
    # longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
    #                                       int(max(train_df['longitude'])),
    #                                       resolution_in_degrees))
    # longitude_bucket_col = tf.feature_column.bucketized_column(longitude_num_col, longitude_boundaries)     # Bucket
    # # Create a feature cross (one-hot indicator, not just multiplication) of latitude and longitude
    # latitude_x_longitude_col = tf.feature_column.crossed_column([latitude_bucket_col, longitude_bucket_col],
    #                                                             hash_bucket_size=100)
    # latitude_x_longitude_indicator_col = tf.feature_column.indicator_column(latitude_x_longitude_col)   # One-hot
    # feature_columns.append(latitude_x_longitude_indicator_col)
    # Convert list of feature columns into a layer for model
    my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    ############################################################################
    # Following section will be re-run empirically and tuned

    # Set proportion of training set to split off as validation set
    my_validation_split = 0.2
    # Tune the hyperparameters
    my_learning_rate = 0.001
    my_n_epochs = 20
    my_batch_size = 100
    my_label_name = 'median_house_value_is_high'
    my_classification_threshold = 0.52

    # Establish the metrics the model will measure.
    my_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=my_classification_threshold),
        tf.keras.metrics.Precision(name='precision', thresholds=my_classification_threshold),
        tf.keras.metrics.Recall(name="recall", thresholds=my_classification_threshold),
        tf.keras.metrics.AUC(name='auc', num_thresholds=100)
    ]

    # Use our modeling functions
    my_model = build_model(my_learning_rate, my_feature_layer, binary=True, metrics=my_metrics)
    epochs, hist_df, weight, bias = \
        train_model(my_model, train_df_norm, my_label_name, my_n_epochs, my_batch_size, my_validation_split)
    print(f"\nYour feature: DISABLED FOR MULTIPLE FEATURES"
          f"\nYour label: {my_label_name}"
          f"\nThe learned weight(s) for your model: {weight}"
          f"\nThe learned bias for your model: {bias}")

    # Use our visualization functions
    # plot_the_model(weight, bias, train_df, my_feature, my_label)
    # plot_the_loss_curve(epochs, rmse_train, rmse_val)
    plot_metrics_curve(epochs, hist_df, my_metrics)

    # Sanity check a random batch of predictions
    # predict_random_batch(my_model, train_df_norm, my_label_name, 15)

    # Ultimate judgment: use test set to evaluate the model
    test_features_df = test_df_norm.copy()
    test_label_ser = test_features_df.pop(my_label_name)
    results = my_model.evaluate(dict(test_features_df), test_label_ser, batch_size=my_batch_size)
