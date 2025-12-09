# Pipeline de limpieza y  procesado de información 
# importar  librerias y  frameworks necesarios para análisis
import pandas as pd
import numpy as  np

# librerias  para  ajuste de sistema
import sys
import os
sys.path.append(os.path.abspath("../"))


# carga de modulos 
from src.utils.utils import limpiar_negativos
#path de almacenamiento de arvhico 
path_df_processing = r'C:\Users\dioct\OneDrive\Desktop\repos\demand_forcasting_and_promotions\Data\processed\data_xyz_df_clean.csv'

# carga de  datos crudos desde el data/raw
data = pd.read_csv(r'C:\Users\dioct\OneDrive\Desktop\repos\demand_forcasting_and_promotions\Data\raw\data_xyz_foods.csv') 

# eliminación de valores ceros sobre variable 'Price'
df = data[data.Price > 0].copy()
print(f'Dimensiones del df original : {data.shape[0]} registros ')
print(f'Dimensiones del df limpiado : {df.shape[0]} registros ')

# Ajustar  variable Date de string a formato datetime
df['Date'] = pd.to_datetime(df['Date'])

# control de valores negativos de las columnas numericas Price y Quantity 
cols = ['Price', 'Quantity' ]
df = limpiar_negativos(df , cols)

# eliminación de duplicados 
df_sin_duplicados = df.drop_duplicates().copy()
# guardar df limpiado y procesado
df_sin_duplicados.to_csv(path_df_processing, index= False, sep=',', header=True)


