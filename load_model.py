#load saved keras model
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

model = keras.models.load_model('dnn_model/')
model.summary()
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
    print(len(dataset.columns))
    dataset = pd.get_dummies(
        dataset, columns=['location'], prefix='', prefix_sep='')
    impute_columns = dataset.columns
    for column in impute_columns:
        dataset[column] = dataset[column].fillna(dataset[column].mean())
    return (dataset)

dataset = clean_data(file)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Transpose location
train_dataset.describe().transpose()

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy() 
train_labels = train_features.pop('new_deaths_per_million')
test_labels = test_features.pop('new_deaths_per_million')

# Predict
test_predictions = model.predict(test_features).flatten()

for actual, predicted in zip(test_labels, test_predictions):
    #diff is percent error
    diff = ((actual - predicted)/actual) * 100
    print(f"Actual: {actual} ===== Test: {predicted} ===== Difference: {diff}" )