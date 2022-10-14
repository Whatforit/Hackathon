
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
np.set_printoptions(precision=3, suppress=True)


file = 'owid-covid-data-large.csv'
target = 'total_deaths_per_million'
threshhold = 4
node_scalar = 5
epochs = 1000

def clean_data(file):

    raw_dataset = pd.read_csv(file, na_values='?', comment='\t', sep=',')
    dataset = raw_dataset.copy()
    #dataset = dataset.iloc[1:, :]
    dataset.isna().sum()
    #dataset = dataset.dropna(axis=1, how='all')
    #dataset = dataset.dropna(axis=0, thresh=19)
    dataset = dataset.drop(
        ['iso_code', 'continent', 'date', 'total_cases','new_cases', 'new_cases_smoothed',	'total_deaths',	'new_deaths', 'new_deaths_smoothed', 'new_cases_smoothed_per_million',	'new_deaths_smoothed_per_million',	'icu_patients',	'hosp_patients', 'weekly_icu_admissions',	'weekly_hosp_admissions', 'total_tests', 'new_tests', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',	'tests_per_case', 'tests_units', 'total_vaccinations',	'people_vaccinated', 'people_fully_vaccinated',	'total_boosters',	'new_vaccinations', 'new_vaccinations_smoothed', 'new_vaccinations_smoothed_per_million',	'new_people_vaccinated_smoothed',	'new_people_vaccinated_smoothed_per_hundred',	'excess_mortality_cumulative_absolute',	'excess_mortality_cumulative',	'excess_mortality']
 , axis=1)
    dataset = pd.get_dummies(
        dataset, columns=['location'], prefix='', prefix_sep='')
    dataset = dataset.dropna(subset=[target])
    dataset = dataset.dropna(thresh=dataset.shape[1]-threshhold, axis=0)
    impute_columns = dataset.columns
    for column in impute_columns:
        dataset[column] = dataset[column].fillna(dataset[column].mean())
    return (dataset)

dataset = clean_data(file)
features = len(dataset.index)
print(f'Features: {features}')

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Transpose location
train_dataset.describe().transpose()

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy() 
train_labels = train_features.pop(target)
test_labels = test_features.pop(target)

#
print(train_dataset.describe().transpose()[['mean', 'std']])


# Create normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print("Normalizer")
print(normalizer.mean.numpy())

nodes = math.ceil(features / (node_scalar * (len(dataset.columns) + 1)) )

print(f"Nodes: {nodes}")
# Build the model
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(nodes, activation='relu'),
      #layers.Dense(nodes, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

# Compile the model
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# Train the model
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=epochs)



#plot_loss(history)

test_results = {}

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# Predict
test_predictions = dnn_model.predict(test_features).flatten()
differences = []
def test_model(test_labels, test_predictions):
    for actual, predicted in zip(test_labels, test_predictions):
        diff = abs(actual - predicted)
        print(f"Actual: {actual} ===== Predicted: {predicted} ===== Difference: {diff}")
        differences.append(diff)
        if diff > 1000:
            print(f"HUGE DIFFERENCE: {diff}")
  # Calculate average percent error
test_model(test_labels, test_predictions)
average_diff = sum(differences)/len(differences)
print(f"Average difference: {average_diff}")
print(f"Max difference: {max(differences)}")
print(f"Min difference: {min(differences)}")


# Save the model
dnn_model.save('dnn_model')






