import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


def build_model(learning_rate):
    """ Create and compile a simple linear regression model
    :param learning_rate: chosen learning rate to pass into Keras optimizer
    :return: compiled TensorFlow Keras model
    """
    # Most simple Keras models are sequential;
    # Sequential model contains one or more layers
    model = tf.keras.models.Sequential()

    # Describe topography of model
    #   - Simple linear regression model: single node in a single layer
    one_node_layer = tf.keras.layers.Dense(units=1, input_shape=(1,))
    model.add(one_node_layer)

    # Compile model topography into efficient TF code
    # Configure training to minimize model's mean squared error
    #   - RMSprop is gradient descent method based on rprop and similar to Adagrad
    rmsprop_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)
    mse_metric = tf.keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=rmsprop_optimizer,
                  loss="mean_squared_error",
                  metrics=[mse_metric])

    return model


def train_model(model, df, feature, label, batch_size, epochs):
    """ Train model by feeding it data
    :param model: compiled Tensorflow Keras model
    :param df:
    :param feature:
    :param label:
    :param batch_size:
    :param epochs:
    :return:
    """
    # Feed feature values and label values to model;
    # Model will train for specified number of epochs, gradually learning how
    # feature values relate to label values
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias
    trained_weight = model.get_weights()[0].item()
    trained_bias = model.get_weights()[1].item()

    # Gather snapshot of each epoch
    hist_df = pd.DataFrame(history.history)
    epochs = history.epoch  # list of epochs, stored separately from rest of history

    # Specifically gather model's root mean squared error at each epoch
    rmse = hist_df["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


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


def plot_the_loss_curve(epochs, rmse):
    """ Plot the loss curve, which shows loss vs. epoch
    :param epochs:
    :param rmse:
    :return:
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Label axes
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Root Mean Squared Error")

    # Create line of x-axis number of epoch vs. y-axis loss
    ax.plot(epochs, rmse, label="Loss")
    ax.legend()
    ax.set_ylim([rmse.min()*0.97, rmse.max()])

    # Render figure
    plt.show()


###############################################################################

if __name__ == '__main__':
    # Import California Housing Dataset
    training_df = pd.read_csv(
        filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    training_df["median_house_value"] /= 1000.0     # Scale the label
    training_df.head()  # Print first rows
    training_df.describe()  # Print stats
    training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]    # Synthetic feature
    training_df.corr()  # Print correlation matrix

    # Specify our chosen feature and label
    my_feature = "median_income"
    my_label = "median_house_value"  # Median value of a house on a specific city block
    # my_synthetic_feature = "rooms_per_person"

    # Tune the hyperparameters
    my_learning_rate = 0.06
    my_batch_size = 30
    my_n_epochs = 24

    # Use our modeling functions
    my_model = build_model(my_learning_rate)
    weight, bias, epochs_hist, rmse_hist = train_model(my_model, training_df, my_feature, my_label,
                                                       my_batch_size, my_n_epochs)
    print(f"\nYour feature: {my_feature}"
          f"\nYour label: {my_label}"
          f"\nThe learned weight for your model is {weight:.4f}"
          f"\nThe learned bias for your model is {bias:.4f}")

    # Use our visualization functions
    plot_the_model(weight, bias, training_df, my_feature, my_label)
    plot_the_loss_curve(epochs_hist, rmse_hist)

    # House prediction function
    def predict_house_values(n, feature, label):
        """Predict house values based on a feature."""
        batch = training_df[feature][10000:10000 + n]
        predicted_values = my_model.predict_on_batch(x=batch)
        print("feature   label          predicted")
        print("  value   value          value")
        print("          in thousand$   in thousand$")
        print("--------------------------------------")
        for i in range(n):
            print("%5.2f %6.1f %15.1f" % (training_df[feature][10000 + i],
                                          training_df[label][10000 + i],
                                          predicted_values[i][0]))
    predict_house_values(15, my_feature, my_label)
