from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

population = {
    
#     Colombia
    "huila_2016-2017.csv" : 1154804,
    "santander_2016-2017.csv" : 2061095,
    "santander_norte_2016-2017.csv" : 1355723,
    "tolima_2016-2017.csv" : 1408274,
    "valle_cauca_2016-2017.csv" : 4613377,
    
#     Brazil
    "Bahia_2016-2017.csv": 15344447,
    "MatoGrosso_2016-2017.csv": 3344544,
    "MinasGerais_2016-2017.csv": 21119536,
    "RioDeJaneiro_2016-2017.csv": 16718956,
    "SaoPaulo_2016-2017.csv": 45094866,

    # Mexico
    "Chiapas_2016-2017.csv": 5217908,
    "Guerrero_2016-2017.csv": 3533251,
    "NuevoLeon_2016-2017.csv": 5119504,
    "Veracruz_2016-2017.csv": 8112505,
    "Yucatan_2016-2017.csv": 2097175,
}


def loadData(country):
  states = os.listdir("drive/My Drive/ziknet-trends-rolling/data/{}/processed_data".format(country))
  
  df_dict = {}
  for state in states:
    df_dict[state] = pd.read_csv("drive/My Drive/ziknet-trends-rolling/data/{}/processed_data/{}".format(country, state), index_col=0)
  return df_dict

def series_to_supervised(df, outputColumn, n_in=1, n_out=1, dropnan=True):
  n_vars = df.shape[1]
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [("{}(t-{})".format(col, i)) for col in df.columns]
  
  # Append next observation[outputColumn] at n_out obs
  cols.append(df[outputColumn].shift(-n_out+1))
  names+=[outputColumn + "(t+{})".format(n_out-1)]

  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

def getXY(dataset, state, weeksAhead):
    dataset[["Searches"]] /= 100
    n_features = dataset.shape[1]

    scale = population[state]
    dataset[["Cases"]] = dataset[["Cases"]].apply(lambda x: x*100000/scale, axis=1)
    
    n_weeks = 4
    reframed = series_to_supervised(dataset, "Cases",  n_weeks, weeksAhead)
    values = reframed.values
    
    totalFeatures = values.shape[1]

    x,y = values[:, :totalFeatures-1], values[:, totalFeatures-1] #Y is the last column, X is all the previous columns 

    x = x.reshape((x.shape[0], n_weeks, n_features)) # Reshape as 3-D
    return x, y