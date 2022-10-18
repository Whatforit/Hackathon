import json
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np

model = keras.models.load_model('dnn_model/')
json_countries = open('./json/fun-data.json', 'r')
json_percentiles = open('./json/fun-data-percentile.json', 'r')

countries = json.loads(json.load(json_countries))
percs = json.loads(json.load(json_percentiles))

def gen_json():
  new_countries = {}
  for name, props in countries.items():
    countries[name].pop("total_deaths_per_million")
    cols = [key for key in countries[name]]
    lst = [countries[name].values()]
    real = pd.DataFrame(lst, columns=cols, dtype = float)
    prediction_real = model.predict(real)
    print(prediction_real)
    new_country_dict = {}
    for prop, value in props.items():
      if percs.get(prop) == None: 
        continue
      impute = percs[prop]["90th Percentile"]
      vals2 = countries[name].copy()
      vals2.update({prop: impute})
      cols2 = [key for key in vals2]
      lst2 = [vals2.values()]
      imputed = pd.DataFrame(lst2, columns=cols2, dtype = float)
      prediction_imputed = model.predict(imputed)
      diff = prediction_real - prediction_imputed

      print('updating newcountrydict!')
      new_country_dict.update({prop: {
        'real_val': value,
        'imputed_val': impute,
        'real_deaths': prediction_real,
        'imputed_deaths': prediction_imputed,
        'diff': diff
      }})
    countryvals = {name}
    new_countries.update({name: new_country_dict})
  return new_countries

print(gen_json())

json_countries.close()
json_percentiles.close()