
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import math
np.set_printoptions(precision=3, suppress=True)


file = 'owid-covid-data-large.csv'
target = 'total_deaths_per_million'
threshhold = 2
node_scalar = 4
epochs = 5000


def clean_data(raw_dataset, drop=True, impute=True):

    dataset = raw_dataset.copy()
    dataset.isna().sum()
    drop_horizontally = [
        'people_vaccinated_per_hundred',
        'total_boosters_per_hundred',
        'stringency_index',
        'population_density',
        'median_age',  
        'gdp_per_capita',
        'cardiovasc_death_rate',
        'diabetes_prevalence',
        'female_smokers',
        'male_smokers',
        'hospital_beds_per_thousand',
        'life_expectancy',
        'human_development_index'
        ]
    dataset = dataset.drop(
        ['iso_code',
         'location',
         #'continent',
         'date',
         'total_cases',
         'new_cases',
         'new_cases_smoothed',
         'total_deaths',
         'new_deaths',
         'new_deaths_smoothed',
         'total_cases_per_million',
         'new_cases_smoothed_per_million',
         #'total_deaths_per_million',
         'new_deaths_per_million',
         'new_deaths_smoothed_per_million',
         'reproduction_rate',
         'icu_patients',
         'icu_patients_per_million',
         'weekly_icu_admissions',
         'weekly_icu_admissions_per_million',
         'hosp_patients',
         'hosp_patients_per_million',
         'weekly_hosp_admissions',
         'weekly_hosp_admissions_per_million',
         'total_tests',
         'new_tests',
         'new_tests_smoothed',
         'total_tests_per_thousand',
         'new_tests_per_thousand',
         'new_tests_smoothed',
         'new_tests_smoothed_per_thousand',
         'positive_rate',
         'tests_per_case',
         'tests_units',
         'total_vaccinations',
         'people_vaccinated',
         'people_fully_vaccinated',
         'total_boosters',
         'new_vaccinations',
         'total_vaccinations_per_hundred',
         'people_fully_vaccinated_per_hundred',
         'new_vaccinations_smoothed_per_million',
         'new_people_vaccinated_smoothed',
         'new_people_vaccinated_smoothed_per_hundred',
         'population',
         'extreme_poverty',
         'handwashing_facilities',
         'excess_mortality_cumulative_absolute',
         'excess_mortality_cumulative',
         'excess_mortality',
         'excess_mortality_cumulative_per_million',
         'aged_70_older',
        ], axis=1)
    print(len(dataset.columns))
    dataset = pd.get_dummies(
        dataset, columns=['continent'], prefix='', prefix_sep='')
    if drop:
        dataset = dataset.dropna(subset=[target])
        dataset = dataset.dropna(subset=drop_horizontally)
        print(dataset.loc[:, dataset.isna().any()])
    if impute:
        impute_columns = dataset.columns
        for column in impute_columns:
            dataset[column] = dataset[column].fillna(dataset[column].mean())
    return (dataset)

# Load the data
raw_dataset = pd.read_csv(file, na_values='?', comment='\t', sep=',')

dataset = clean_data(raw_dataset)

features = len(dataset.index)

print(f'Features: {features}')

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop(target)
test_labels = test_features.pop(target)

# Print the model summary
print(train_dataset.describe().transpose()[['mean', 'std']])


# Create normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print("Normalizer")
print(normalizer.mean.numpy())

nodes = math.ceil(features / (node_scalar * (len(dataset.columns) + 1)))

print(f"Nodes: {nodes}")

# Build the model
def build_and_compile_model(norm):

    model = keras.Sequential([
        norm,
        layers.Dense(nodes, activation='relu'), # 1st hidden layer, nodes based on features
        layers.Dense(1) # Output layer
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001)) # Adam optimizer
    return model


# Compile the model
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# Early stopping, stop training if the validation loss doesn't improve. Prevents overfitting
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'auto',
                                                 patience=5,
                                                 restore_best_weights=True)

# Train the model
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=epochs, callbacks=[earlystopping])



differences = []
percent_diff = []
test_results = {}
def test_model(test_labels):
    test_results['dnn_model'] = dnn_model.evaluate(
            test_features, test_labels, verbose=0)
    test_predictions = dnn_model.predict(test_features).flatten() # Predict the test data and flatten the array
    for actual, predicted in zip(test_labels, test_predictions):
        diff = abs(actual - predicted)
        if actual != 0:
            percent_error = (diff / actual) * 100
            percent_diff.append(percent_error)
        print(
            f"Actual: {actual} ===== Predicted: {predicted} ===== Difference: {diff}")
        differences.append(diff)
        if diff > 1000:
            print(f"HUGE DIFFERENCE: {diff}")


test_model(test_labels)

# Print predictions and performance metrics
percent_diff = np.array(percent_diff).mean()
print(f"Average difference: {percent_diff}")
print(f"Max difference: {max(differences)}")
print(f"Min difference: {min(differences)}")


# Save the model
if input("Save model? (y/n): ") == 'y':
    dnn_model.save('models/')
    print("Model saved")
