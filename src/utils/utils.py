# importar  librerias  necesarias
import pandas as pd
import numpy as np

def limpiar_negativos(df, columnas):
    negativos = (df[columnas] < 0).any()
    
    if negativos.any():
        print(" AL validar se encontro el total de valores negativos por columna de :", list(negativos[negativos].index))
        return df[(df[columnas] >= 0).all(axis=1)]
    else:
        print('No hay presencia de valores negativos en el df')
        return df
