
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
np.set_printoptions(precision=3, suppress=True)

data = 'owid-covid-data.csv'

column_names = ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
                'tests_per_case', 'positive_rate', 'tests_units', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy']
raw_dataset = pd.read_csv(data, names=column_names,
                          na_values='?', comment='\t',
                          sep=',')

dataset = raw_dataset.copy()

dataset.isna().sum()
dataset = dataset.dropna(axis=1, how='all')
dataset = dataset.dropna()
dataset = dataset.drop(['iso_code', 'continent', 'date'], axis=1)
dataset = pd.get_dummies(dataset, columns=['location'], prefix='', prefix_sep='')
print(dataset.to_string())

train_dataset = dataset.sample(frac=0.8, random_state=0)

test_dataset = dataset.drop(train_dataset.index)

train_dataset.describe().transpose()


train_features = train_dataset.copy()
test_features = test_dataset.copy()
#train_features=np.asarray(train_features).astype(np.float32)


train_labels = train_features.pop('total_deaths_per_million')
test_labels = test_features.pop('total_deaths_per_million')
#train_labels=np.asarray(train_labels).astype(np.float32)

train_dataset.describe().transpose()[['mean', 'std']]


normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


handwash = np.array(train_features['handwashing_facilities'])

handwash_normalizer = layers.Normalization(input_shape=[1, ], axis=None)

handwash_normalizer.adapt(handwash)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=1,
    validation_split = 0.2)


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [total_deaths_per_million]')
  plt.legend()
  plt.grid(True)


plot_loss(history)

test_results = {}
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=1)

test_predictions = linear_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [total_deaths_per_million]')
plt.ylabel('Predictions [total_deaths_per_million]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=100)


plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=1)

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [deaths]')
plt.ylabel('Predictions [deaths]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [deaths]')
_ = plt.ylabel('Count')


dnn_model.save('covid_dnn_model')





