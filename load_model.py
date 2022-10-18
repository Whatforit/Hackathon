#load saved keras model
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from io import StringIO
import math
import csv

model = keras.models.load_model('dnn_model/')
model.summary()
file = 'owid-covid-data-large.csv'
target = 'total_deaths_per_million'
threshhold = 5
def clean_data(raq_dataset, drop=True, impute=True):

    dataset = raw_dataset.copy()
    #dataset = dataset.iloc[1:, :]
    dataset.isna().sum()
    #dataset = dataset.dropna(axis=1, how='all')
    #dataset = dataset.dropna(axis=0, thresh=19)
    dataset = dataset.drop(
        ['iso_code',
 'continent',
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
 'new_deaths_smoothed_per_million',
 'reproduction_rate',
 'icu_patients',
 'hosp_patients',
 'weekly_icu_admissions',
 'weekly_icu_admissions_per_million',
 'weekly_hosp_admissions',
 'weekly_hosp_admissions_per_million',
 'total_tests',
 'new_tests',
 'new_tests_smoothed',
 'new_tests_smoothed_per_thousand',
 'tests_per_case',
 'tests_units',
 'total_vaccinations',
 'new_vaccinations_smoothed',
 'new_vaccinations_smoothed_per_million',
 'new_people_vaccinated_smoothed',
 'new_people_vaccinated_smoothed_per_hundred',
 'excess_mortality_cumulative_absolute',
 'excess_mortality_cumulative'], axis=1)
    print(f'cols: {len(dataset.columns)}')
    print(dataset.columns)
    dataset = pd.get_dummies(
        dataset, columns=['location'], prefix='', prefix_sep='')
    if drop:
        dataset = dataset.dropna(subset=[target])
        dataset = dataset.dropna(thresh=dataset.shape[1]-threshhold, axis=0)
    if impute:
        impute_columns = dataset.columns
        for column in impute_columns:
            dataset[column] = dataset[column].fillna(dataset[column].mean())
    return (dataset)

raw_dataset = pd.read_csv(file, na_values='?', comment='\t', sep=',')
dataset = clean_data(raw_dataset)

# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

# # Transpose location
# train_dataset.describe().transpose()

# # Split features from labels
# train_features = train_dataset.copy()
# test_features = test_dataset.copy() 
# train_labels = train_features.pop(target)
# test_labels = test_features.pop(target)

# # Predict
# test_predictions = model.predict(test_features).flatten()
# differences = []
# def test_model(test_labels, test_predictions):
#     for actual, predicted in zip(test_labels, test_predictions):
#         diff = abs(actual - predicted)
#         print(f"Actual: {actual} ===== Predicted: {predicted} ===== Difference: {diff}")
#         differences.append(diff)
#         if diff > 1000:
#             print(f"HUGE DIFFERENCE: {diff}")
#   # Calculate average percent error
# test_model(test_labels, test_predictions)
# average_diff = sum(differences)/len(differences)
# print(f"Average difference: {average_diff}")
# print(f"Max difference: {max(differences)}")
# print(f"Min difference: {min(differences)}")
# '''
# names = "iso_code,continent,location,date,total_cases,new_cases,new_cases_smoothed,total_deaths,new_deaths,new_deaths_smoothed,total_cases_per_million,new_cases_per_million,new_cases_smoothed_per_million,total_deaths_per_million,new_deaths_per_million,new_deaths_smoothed_per_million,reproduction_rate,icu_patients,icu_patients_per_million,hosp_patients,hosp_patients_per_million,weekly_icu_admissions,weekly_icu_admissions_per_million,weekly_hosp_admissions,weekly_hosp_admissions_per_million,total_tests,new_tests,total_tests_per_thousand,new_tests_per_thousand,new_tests_smoothed,new_tests_smoothed_per_thousand,positive_rate,tests_per_case,tests_units,total_vaccinations,people_vaccinated,people_fully_vaccinated,total_boosters,new_vaccinations,new_vaccinations_smoothed,total_vaccinations_per_hundred,people_vaccinated_per_hundred,people_fully_vaccinated_per_hundred,total_boosters_per_hundred,new_vaccinations_smoothed_per_million,new_people_vaccinated_smoothed,new_people_vaccinated_smoothed_per_hundred,stringency_index,population,population_density,median_age,aged_65_older,aged_70_older,gdp_per_capita,extreme_poverty,cardiovasc_death_rate,diabetes_prevalence,female_smokers,male_smokers,handwashing_facilities,hospital_beds_per_thousand,life_expectancy,human_development_index,excess_mortality_cumulative_absolute,excess_mortality_cumulative,excess_mortality,excess_mortality_cumulative_per_million"
# names_csv = names.split(',')
# print(names_csv)
# point="AFG,Asia,Afghanistan,2020-03-11,11.0,3.0,0.857,,,,0.274,0.075,0.021,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,27.78,40099462.0,54.422,18.6,2.581,1.337,1803.987,,597.029,9.59,,,37.746,0.5,64.83,0.511,,,,"
# point_csv = [point.split(',')]
# pointdf = pd.DataFrame(point_csv, columns=names_csv)
# cleandata = clean_data(pointdf, drop=True, impute=False)
# #print(cleandata.to_string())
# point_test_prediction = model.predict(cleandata).flatten()
# print(point_test_prediction)
# '''