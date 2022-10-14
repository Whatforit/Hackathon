import pandas as pd

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
x = dataset.drop(['new_deaths_per_million'], axis=1)
y = dataset['new_deaths_per_million']


x_train_full, x_test, y_train_full, y_test = train_test_split(x, y)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_full, y_train_full)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



input = tf.keras.layers.Input(shape = x_train.shape[1:])
hidden1 = tf.keras.layers.Dense(300, activation='relu')(input)
hidden2 = tf.keras.layers.Dense(300, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(300, activation='relu')(hidden2)
hidden4 = tf.keras.layers.Concatenate()([input, hidden3])
hidden5 = tf.keras.layers.Dense(300, activation='relu')(hidden4)
concat = tf.keras.layers.Concatenate()([input, hidden5])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.models.Model(inputs=[input], outputs=[output])

model.compile(
    loss=rmse,
    optimizer=Adam(),
    metrics=[rmse]
)

history4 = model.fit(x_train, y_train, epochs=20000,
                      validation_data=(x_valid, y_valid))

model.evaluate(x_test, y_test)
