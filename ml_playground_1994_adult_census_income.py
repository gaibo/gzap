import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
# from ml_playground import build_model, train_model, plot_metrics_curve
from facets_overview.feature_statistics_generator import FeatureStatisticsGenerator
import base64
# from IPython.display import display, HTML
import webbrowser
import seaborn as sns
import matplotlib as mpl
mpl.use('Qt5Agg')   # matplotlib 3.5 changed default backend and PyCharm freezes

# Download data to local .keras/datasets/ cache (returns file path)
train_csv = tf.keras.utils.get_file('adult.data',
                                    'https://download.mlcc.google.com/mledu-datasets/adult_census_train.csv')
test_csv = tf.keras.utils.get_file('adult.test',
                                   'https://download.mlcc.google.com/mledu-datasets/adult_census_test.csv')
DATA_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "gender",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
# Read data from local directory
# NOTE: data is very raw, here are what the arguments do:
#   - "names" assigns column names; pandas default to using first row of data as column names
#   - "sep" specifies to anticipate whitespace around commas; otherwise you get ' Married-civ-spouse', etc.
#   - "engine" appears unnecessary - complex, regex sep forces 'python', which is most feature-complete but slowest;
#     however, specifying avoids ParserWarning as default appears to be 'c' engine
#   - "na_values" specifies additional strings to interpret as NaN
#   - "skiprows" specifies to skip row 0; test data has a first row with text "|1x3 Cross validator"
train_df = pd.read_csv(train_csv, names=DATA_COLUMNS, sep=r'\s*,\s*', engine='python', na_values='?')
test_df = pd.read_csv(test_csv, names=DATA_COLUMNS, sep=r'\s*,\s*', engine='python', na_values='?', skiprows=[0])
# Strip trailing periods mistakenly included only in UCI (University of California, Irvine ML repo) test dataset
test_df['income_bracket'] = test_df['income_bracket'].str.rstrip('.')   # .rstrip() as in "right", i.e. trailing

################################################################################
# Google Facets Dataset Exploration

# Visualize data in Facets Overview
# NOTE: to get this code working, I had to
#       1) downgrade numpy to 1.22.4, when np.bool was still a dtype
#       2) downgrade protobuf to 3.20.0
fsg = FeatureStatisticsGenerator()
dataframes = [
    {'name': 'census_train', 'table': train_df}
]
census_proto = fsg.ProtoFromDataFrames(dataframes)   # JSON-like FSG object
census_proto_str = base64.b64encode(census_proto.SerializeToString()).decode('utf-8')   # Convert FSG to denser encoding
OVERVIEW_HTML_TEMPLATE = """\
<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
<facets-overview id="elem"></facets-overview>
<script>
    document.querySelector("#elem").protoInput = "{protostr}";
</script>
"""
overview_html = OVERVIEW_HTML_TEMPLATE.format(protostr=census_proto_str)
# display(HTML(overview_html))
overview_html_local_filename = 'facets_overview_census_1994.html'
with open(overview_html_local_filename, 'w') as file:
    file.write(overview_html)
webbrowser.open(overview_html_local_filename)

# Visualize data in Facets Dive
SAMPLE_SIZE = 5000
census_sample_json = train_df.sample(SAMPLE_SIZE).to_json(orient='records')     # 'records' is list-like, column->value
DIVE_HTML_TEMPLATE = """\
<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
<facets-dive id="elem"></facets-dive>
<script>
    var data = {jsonstr};
    document.querySelector("#elem").data = data;
</script>
"""
dive_html = DIVE_HTML_TEMPLATE.format(jsonstr=census_sample_json)
dive_html_local_filename = 'facets_dive_census_1994.html'
with open(dive_html_local_filename, 'w') as file:
    file.write(dive_html)
webbrowser.open(dive_html_local_filename)

################################################################################
# Machine Learning

# Drop all NaNs examples, since we profiled the data and decided that was fine
clean_train_df = train_df.dropna(how='any')
# Split off income_bracket as label and leave the rest as features
train_labels = clean_train_df['income_bracket'].eq('>50K')    # NOTE: True/False, not 1/0
train_features = dict(clean_train_df.drop('income_bracket', axis=1))

# Don't know full range of possible values, so map each category feature string to integer ID
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)

# Know possible values for categories, can be more explicit
gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['Female', 'Male'])
race = tf.feature_column.categorical_column_with_vocabulary_list(
    'race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Black', 'Other'])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
                  'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors',
                  'Masters', 'Prof-school', 'Doctorate'])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                       'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov',
                  '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

# Numeric feature columns (use raw value, instead of having to create map between string and ID)
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
fnlwgt = tf.feature_column.numeric_column('fnlwgt')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

# # List of selected features; special handling for GENDER SUBGROUP
# variables = [occupation, native_country, education, workclass, relationship, age_buckets]
# subgroup_variables = [gender]
# feature_columns = variables + subgroup_variables

# Convert high-dimensional categorical features into low-dimensional/dense real-valued embedding vectors
# NOTE: use indicator_column (one-hot encoding) and embedding_column (sparse to dense)
deep_columns = [
    tf.feature_column.embedding_column(occupation, dimension=8),
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.indicator_column(age_buckets)
]

# Define DNN model
HIDDEN_UNITS_LAYER_01 = 128
HIDDEN_UNITS_LAYER_02 = 64
LEARNING_RATE = 0.1
L1_REGULARIZATION_STRENGTH = 0.001
L2_REGULARIZATION_STRENGTH = 0.001
tf.random.set_seed(512)     # RANDOM_SEED = 512
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]
regularizer = tf.keras.regularizers.l1_l2(
    l1=L1_REGULARIZATION_STRENGTH, l2=L2_REGULARIZATION_STRENGTH)
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(deep_columns),
    tf.keras.layers.Dense(
        HIDDEN_UNITS_LAYER_01, activation='relu', kernel_regularizer=regularizer),
    tf.keras.layers.Dense(
        HIDDEN_UNITS_LAYER_02, activation='relu', kernel_regularizer=regularizer),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
])
model.compile(optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

# Fit DNN model to training dataset
EPOCHS = 10
BATCH_SIZE = 500    # TODO: investigate batch size more
model.fit(x=train_features, y=train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate DNN model performance on test dataset
clean_test_df = test_df.dropna(how='any')
test_labels = clean_test_df['income_bracket'].eq('>50K')    # NOTE: True/False, not 1/0; technically equiv in Python
test_features = dict(clean_test_df.drop('income_bracket', axis=1))
model.evaluate(x=test_features, y=test_labels)


CONFUSION_MATRIX_2X2_LABELS = np.asarray([['True Positives', 'False Positives'],
                                          ['False Negatives', 'True Negatives']])


def plot_confusion_matrix(confusion_matrix, class_names, subgroup_title=None, figsize=(8, 6)):
    """ Create 2x2 confusion matrix using Seaborn heatmap and model metrics
    :param confusion_matrix: 2x2 array or DataFrame with (count) values in format:
                             True Positive | False Positive
                             False Negative | True Negative
    :param class_names: array or list of length 2 of string names in format:
                        Positive class name | Negative class name
    :param subgroup_title: string name of targeted dataset to add to figure title; set None if not relevant
    :param figsize: matplotlib.figure size in 100s of pixels (horizontal, vertical)
    :return: matplotlib.figure
    """
    if isinstance(confusion_matrix, pd.DataFrame):
        confusion_matrix = confusion_matrix.to_numpy()  # Ensure array
    # Create display labels in format:
    # {TP_count}\nTrue Positive | {FP_count}\nFalse Positive
    # {FN_count}\nFalse Negative | {TN_count}\nTrue Negative
    formatted_labels = (np.asarray([f'{value:g}\n{string}' for value, string in
                                    zip(confusion_matrix.flatten(), CONFUSION_MATRIX_2X2_LABELS.flatten())])
                        .reshape(2, 2))
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)
    # Plot heatmap with Seaborn
    # NOTE: fmt='' needed to assert no formatting to our pre-formatted label strings
    # NOTE: ax automatically used by Seaborn; fig will have 2 axes - heatmap and colorbar
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm, annot=formatted_labels, fmt='', linewidths=2.0, cmap=sns.color_palette("GnBu_d"))
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')    # Horiz instead of vert
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')   # Lower left to upper right
    # Label axes
    if subgroup_title is None:
        subgroup_title = 'All'
    ax.set_title('Confusion Matrix for Performance Across: ' + subgroup_title)  # NOTE: ax == heatmap
    ax.set_ylabel("Predictions")    # Want model "positives" on top row, "negatives" on bottom
    ax.set_xlabel("References")
    # Adjust text fit
    fig.tight_layout()
    # Render figure
    plt.show()


# Subgroup confusion matrix sanity check
CATEGORY = 'gender'
SUBGROUP = 'Female'
subgroup_filter = clean_test_df.loc[clean_test_df[CATEGORY] == SUBGROUP]
# Filter labels and features for subgroup
subgroup_test_labels = subgroup_filter['income_bracket'].eq('>50K')    # NOTE: True/False, not 1/0
classes = ['Over $50K', 'Under $50K']
subgroup_test_features = dict(subgroup_filter.drop('income_bracket', axis=1))
# Evaluate model on just subgroup
subgroup_results = model.evaluate(x=subgroup_test_features, y=subgroup_test_labels, verbose=0)
# Construct confusion matrix from model evaluation metrics
subgroup_cm = np.array([[subgroup_results[1], subgroup_results[2]],
                        [subgroup_results[3], subgroup_results[4]]])
subgroup_performance_metrics = {
    'ACCURACY': subgroup_results[5],
    'PRECISION': subgroup_results[6],
    'RECALL': subgroup_results[7],
    'AUC': subgroup_results[8]
}
performance_df = pd.DataFrame(subgroup_performance_metrics, index=[SUBGROUP])
# Output visuals
plot_confusion_matrix(subgroup_cm, classes, SUBGROUP)
print(performance_df)
