# from matplotlib import pyplot as plt
import tensorflow as tf
from ml_playground import build_model, train_model, plot_metrics_curve

# Load data from Keras dataset
# NOTE: MNIST x data is stack of 28x28 pixel maps where each pixel is integer [0, 255]
#       on a gray scale where 0 represents white and 255 represents black.
#       MNIST y data is rater identification of each image as integer [0, 9]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Plot image
# plt.imshow(x_train[2916])

# Normalize
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Since MNIST input data is 2D, need to think differently about model topography
# Feature layer - flatten 28x28 into 1-dimensional array of 784
flatten_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
# Hidden layers + dropout regularization
hidden_layer_1 = tf.keras.layers.Dense(units=256, activation='relu')
hidden_layer_2 = tf.keras.layers.Dense(units=128, activation='relu')
dropout_reg_layer = tf.keras.layers.Dropout(rate=0.4)
hidden_layers_list = [hidden_layer_1, dropout_reg_layer, hidden_layer_2]
# Output layer - choose between 10 possible output values [0, 9]
output_layer = tf.keras.layers.Dense(units=10, activation='softmax')

# Flesh out model details
optimizer = 'Adam'
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']

################################################################################

# Hyperparameters
learning_rate = 0.003
n_epochs = 50
batch_size = 4000
validation_split = 0.2

# Establish model topography
my_model = build_model(learning_rate, flatten_layer, hidden_layers_list, output_layer,
                       optimizer=optimizer, loss=loss, metrics=metrics)

# Train model on normalized training set
epochs_list, hist_df, _, _ = train_model(my_model, x_train_normalized, y_train, n_epochs, batch_size, validation_split)

# Visualize with metrics
plot_metrics_curve(epochs_list, hist_df, metrics=metrics)

# Evaluate against test set
print("\nEvaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

# FINAL RESULTS
# Dropout layer location does matter:
# Dropout, hidden 1, hidden 2 has highest loss but best test accuracy (from less overfitting)
# Hidden 1, dropout, hidden 2 has medium loss, medium accuracy
# Hidden 1, hidden 2, dropout has lowest loss but worst test accuracy
# Does this simply mean I should boost dropout higher, since there's overfitting?
