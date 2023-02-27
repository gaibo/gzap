import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
tf.keras.backend.set_floatx('float32')


def build_model(learning_rate, feature_layer=None):
    """ Create and compile a simple linear regression model
    :param learning_rate: chosen learning rate to pass into Keras optimizer
    :param feature_layer: tf.keras.layers.DenseFeatures()
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
    # Add one linear layer to yield simple linear regressor
    one_node_layer = tf.keras.layers.Dense(units=1, input_shape=(1,))
    model.add(one_node_layer)

    # Compile model topography into efficient TF code
    # Configure training to minimize model's mean squared error
    #   - RMSprop is gradient descent method based on rprop and similar to Adagrad
    rmsprop_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)
    rmse_metric = tf.keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=rmsprop_optimizer,
                  loss="mean_squared_error",
                  metrics=[rmse_metric])

    return model


def train_model(model, dataset_df, label_name, n_epochs, batch_size=None, validation_split=0.2):
    """ Train model by feeding it data
    :param model: compiled Tensorflow Keras model
    :param dataset_df:
    :param label_name: string name of column in dataset_df to use as label;
                       don't need feature names because we pass rest of dataset_df to model, and
                       we assume model has feature columns which already define their names
    :param n_epochs:
    :param batch_size:
    :param validation_split:
    :return:
    """
    # Feed feature values and label values to model;
    # Model will train for specified number of epochs, gradually learning how
    # feature values relate to label values
    features_df = dataset_df.copy()     # Will modify copy to become just features
    label_ser = features_df.pop(label_name)
    history = model.fit(x=dict(features_df),
                        y=label_ser,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_split=validation_split)

    # Gather the trained model's weight and bias
    # NOTE: .item() can extract number from np.array wrapper
    trained_weight = model.get_weights()[0]     # Can be multiple weights with multiple features
    trained_bias = model.get_weights()[1].item()

    # Gather snapshot of each epoch
    history_df = pd.DataFrame(history.history)
    epochs = history.epoch  # list of epochs, stored separately from rest of history

    # Specifically gather model's root mean squared error at each epoch
    rmse_training = history_df["root_mean_squared_error"]
    rmse_validation = history_df["val_root_mean_squared_error"]

    return epochs, rmse_training, rmse_validation, history_df, trained_weight, trained_bias


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


def plot_the_loss_curve(epochs, training_loss, validation_loss):
    """ Plot the loss curve, which shows loss vs. epoch
    :param epochs:
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
    ax.plot(epochs[1:], training_loss[1:], label="Training Loss")
    ax.plot(epochs[1:], validation_loss[1:], label="Validation Loss")
    ax.legend()
    merged_loss_values = pd.concat([training_loss, validation_loss])
    highest_loss, lowest_loss = max(merged_loss_values), min(merged_loss_values)
    delta = highest_loss - lowest_loss
    ax.set_ylim([lowest_loss-delta*0.05, highest_loss+delta*0.05])

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


###############################################################################

if __name__ == '__main__':
    # Import California Housing Dataset
    train_df = pd.read_csv(
        filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    test_df = pd.read_csv(
        filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
    DOLLAR_SCALE_FACTOR = 1000.0    # Scale house value label to be more readable/usable
    train_df["median_house_value"] /= DOLLAR_SCALE_FACTOR
    test_df["median_house_value"] /= DOLLAR_SCALE_FACTOR
    train_df.head()  # Print first rows
    train_df.describe()  # Print stats
    train_df.corr()  # Print correlation matrix
    # Manually shuffle data in training set so validation set isn't biased
    train_df = train_df.sample(frac=1)
    # Set proportion of training set to split off as validation set
    my_validation_split = 0.2

    # Specify our chosen feature and label
    # my_feature = "median_income"    # Median income on a specific city block
    # my_label = "median_house_value"     # Median value of a house on a specific city block
    feature_columns = []    # Initialize list of feature columns
    latitude_feat = tf.feature_column.numeric_column('latitude')    # Numerical feature
    feature_columns.append(latitude_feat)
    longitude_feat = tf.feature_column.numeric_column('longitude')  # Numerical feature
    feature_columns.append(longitude_feat)
    fp_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)   # Floating point feature layer

    # Tune the hyperparameters
    my_learning_rate = 0.05
    my_n_epochs = 30
    my_batch_size = 100
    my_label_name = 'median_house_value'

    # Use our modeling functions
    my_model = build_model(my_learning_rate, fp_feature_layer)
    epochs_list, rmse_train, rmse_val, hist_df, weight, bias = \
        train_model(my_model, train_df, my_label_name, my_n_epochs, my_batch_size, my_validation_split)
    print(f"\nYour feature: DISABLED FOR MULTIPLE FEATURES"
          f"\nYour label: {my_label_name}"
          f"\nThe learned weight(s) for your model: {weight}"
          f"\nThe learned bias for your model: {bias}")

    # Use our visualization functions
    # plot_the_model(weight, bias, train_df, my_feature, my_label)
    plot_the_loss_curve(epochs_list, rmse_train, rmse_val)

    # Sanity check a random batch of predictions
    predict_random_batch(my_model, train_df, my_label_name, 15)

    # # Ultimate judgment: use test set to evaluate the model
    # x_test, y_test = test_df[my_feature], test_df[my_label]
    # results = my_model.evaluate(x_test, y_test, batch_size=my_batch_size)
