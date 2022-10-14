
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
np.set_printoptions(precision=3, suppress=True)


file = 'owid-covid-data-large.csv'

def clean_data(file):

    column_names = ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients', 'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations', 'people_vaccinated',
                    'people_fully_vaccinated', 'total_boosters', 'new_vaccinations', 'new_vaccinations_smoothed', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million', 'new_people_vaccinated_smoothed', 'new_people_vaccinated_smoothed_per_hundred', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index', 'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million']
    raw_dataset = pd.read_csv(file, na_values='?', comment='\t', sep=',')
    dataset = raw_dataset.copy()
    #dataset = dataset.iloc[1:, :]
    dataset.isna().sum()
    #dataset = dataset.dropna(axis=1, how='all')
    #dataset = dataset.dropna(axis=0, thresh=19)
    dataset = dataset.drop(
        ['iso_code', 'continent', 'date', 'total_cases','new_cases', 'new_cases_smoothed',	'total_deaths',	'new_deaths', 'new_deaths_smoothed', 'new_cases_smoothed_per_million',	'new_deaths_smoothed_per_million',	'icu_patients',	'hosp_patients', 'weekly_icu_admissions',	'weekly_hosp_admissions', 'total_tests', 'new_tests', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',	'tests_per_case', 'tests_units', 'total_vaccinations',	'people_vaccinated', 'people_fully_vaccinated',	'total_boosters',	'new_vaccinations', 'new_vaccinations_smoothed', 'new_vaccinations_smoothed_per_million',	'new_people_vaccinated_smoothed',	'new_people_vaccinated_smoothed_per_hundred',	'excess_mortality_cumulative_absolute',	'excess_mortality_cumulative',	'excess_mortality']
 , axis=1)
    print(dataset.columns)
    dataset = pd.get_dummies(
        dataset, columns=['location'], prefix='', prefix_sep='')
    impute_columns = dataset.columns
    for column in impute_columns:
        dataset[column] = dataset[column].fillna(dataset[column].mean())
    return (dataset)

dataset = clean_data(file)

train_dataset = dataset.sample(frac=0.8, random_state=0)


test_dataset = dataset.drop(train_dataset.index)


train_dataset.describe().transpose()



train_features = train_dataset.copy()
test_features = test_dataset.copy()
#train_features=np.asarray(train_features).astype(np.float32)

print(train_features.tail())


train_labels = train_features.pop('total_deaths_per_million')
test_labels = test_features.pop('total_deaths_per_million')
#train_labels=np.asarray(train_labels).astype(np.float32)
print(train_labels)


train_dataset.describe().transpose()[['mean', 'std']]



normalizer = tf.keras.layers.Normalization(axis=-1)


normalizer.adapt(np.array(train_features))

print("Normalizer")
print(normalizer.mean.numpy())


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())



def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [total_deaths_per_million]')
  plt.legend()
  plt.grid(True)



#plot_loss(history)


test_results = {}
'''
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


test_predictions = linear_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [total_deaths_per_million]')
plt.ylabel('Predictions [total_deaths_per_million]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
'''


def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])
model4 = keras.Sequential([
    tf.keras.layers.Input(shape = X_train.shape[1:]),
    layers.Dense(300, activation='tanh'),
    layers.Dense(300, activation='tanh'),
    layers.Dense(300, activation='tanh'),
    layers.Dense(300, activation='tanh'),
    layers.Dense(300, activation='tanh'),
    layers.Dense(300, activation='tanh'),
    layers.Dense(1)
])
  model.compile(loss='rmse',
                optimizer=tf.keras.optimizers.Adam(0.001), metricts=['rmse'])
  return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=100)



plot_loss(history)


test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)


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



dnn_model.save('dnn_model')






