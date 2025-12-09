# Pipeline de l creación nuevas caracteristicas
# importar  librerias y  frameworks necesarios para análisis
import pandas as pd
import numpy as  np

# librerias  para  ajuste de sistema
import sys
import os
sys.path.append(os.path.abspath("../"))

# carga de modulos 
# from src.utils.utils import
#path de almacenamiento de arvhico  con caracteristicas 
path_df_train = r'C:\Users\dioct\OneDrive\Desktop\repos\demand_forcasting_and_promotions\Data\processed\train.csv'
path_df_test = r'C:\Users\dioct\OneDrive\Desktop\repos\demand_forcasting_and_promotions\Data\processed\test.csv'

path_cargue = r'C:\Users\dioct\OneDrive\Desktop\repos\demand_forcasting_and_promotions\Data\processed\data_xyz_df_clean.csv'
#path de almacenamiento de arvhico 


# carga de  datos crudos desde el data/raw
data = pd.read_csv(path_cargue)# Ajustar  variable Date de string a formato datetime
data['Date'] = pd.to_datetime(data['Date'])

# Agrupar los datos  segun fecha,  producto y tienda 
# crear caracteristica de impacto monetario
data['Revenue'] = data.Quantity*data.Price

data_ = data.groupby(['Date', 'ProductID', 'StoreID']). agg(
    cantidad_diaria = ('Quantity', 'sum'),
    importe_diario = ('Revenue', 'sum')).reset_index()
 
# crear  caracteristicas  en unidad temporal 
# feature  creados sobre  Date
data_['anio'] = data_.Date.dt.year
data_['mes_num'] = data_.Date.dt.month
data_['dia_fecha'] = data_.Date.dt.day
data_['dia_semana'] = data_.Date.dt.dayofweek
data_['trimestre'] = data_.Date.dt.quarter
data_['semana_anio'] = data_.Date.dt.isocalendar().week.astype(int)
data_['es_fin_semana'] = data_.dia_semana.isin([5,6])

# crear  caracteristicas  relevantes para analisis de serie temporar
# crear lags de serie temporales diarias sobre  combinación unica de producto y tienda de venta para idenitificar demenda pasada
df_lags = data_.sort_values(by=['StoreID', 'ProductID', 'Date'])

# generar lags  de 1, 7 y 14 dias para  extraer comportamientos de  demanda  pasada
lags = [1,7,14]
# iterar sobre la lista lags para generar las  columnas  nuevas por la informacion de cantidad e importe diario
for lag in lags:
    df_lags[f'cant_diaria_lag{lag}'] = df_lags.groupby(['ProductID', 'StoreID'])['cantidad_diaria'].shift(lag)
    df_lags[f'importe_diaria_lag{lag}'] = df_lags.groupby(['ProductID', 'StoreID'])['importe_diario'].shift(lag)
# crear rolling windows  sobre  los conjuntos de  datos  para identificar patrones temporales  de la demanda e importe  sobre  los 7 y 30 dias anteriores
# esto permite  identificar volatilidad( fluctuaciones temporales)

rolling_windows =  [7, 30]
df_features = df_lags.copy()
for  window in rolling_windows:
      df_features[f'rollingMean_cantidad_diaria_{window}'] = df_features.groupby(['ProductID', 'StoreID'])['cantidad_diaria'].rolling(window=window, min_periods=1).mean().reset_index(level =[0,1], drop=True)
      df_features[f'rollingMean_cantidad_diaria_{window}'] = df_features.groupby(['ProductID', 'StoreID'])['importe_diario'].rolling(window=window, min_periods=1).mean().reset_index(level =[0,1], drop=True)
      df_features[f'rollingSd_cantidad_diaria_{window}'] = df_features.groupby(['ProductID', 'StoreID'])['cantidad_diaria'].rolling(window=window, min_periods=1).std().reset_index(level =[0,1], drop=True)
      df_features[f'rollingSd_cantidad_diaria_{window}'] = df_features.groupby(['ProductID', 'StoreID'])['importe_diario'].rolling(window=window, min_periods=1).std().reset_index(level =[0,1], drop=True)

# control de nulos asociados  a los lags y rolling windows 
df_features = df_features.fillna(0)

# división de conjunto de datos en train y test para el modelo de  demanda 
umbral_fecha = pd.Timestamp('2025-06-30')
train = df_features[df_features.Date <= umbral_fecha].copy()
test = df_features[df_features.Date >umbral_fecha].copy()
print(f' train dataset : {train.shape} \ntest dataset: {test.shape}')

# almacenamiento de conjuntos de datos para entreno y evaluación
train.to_csv(path_df_train, index= False, sep=',', header=True)
test.to_csv(path_df_test, index= False, sep=',', header=True)