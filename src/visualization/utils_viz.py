# importar  librerias  necesarias
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 

# graficos  de  visualización de datos  (EDA)

def plot_count(feature, title, df, size=1):
    """
    Grafica el plot_count de atributos para cada clase categorica
    
    :param df: dataframe  con la data que  sera graficada en el plot_count.
    :param feature: nombre del atributo a ser graficado en el plot_count.
    :param title: titulo del grafico.
    :param size:  tamaño que  se desea que se expanda la grafica de forma horizontal.
    """
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    #total = float(len(df))
    g = sns.countplot(x = df[feature], order = df[feature].value_counts().index, palette='Set3')
    g.set_title("Resultado por cada categoria de {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}'.format(height),
                ha="center") 
    plt.show()

def barplot_per_classes(df, attribute, groupby, title=None, ticks_rotation=0, topn=None, ax=None):
    """
    Grafica el Barplot de atributos para cada clase categorica
    
    :param df: dataframe  con la data que  sera graficada en el barplot.
    :param attribute: nombre del atributo a ser graficado en el barplot.
    :param groupby: nombre del atributo con la clase predictora.
    :param title: titulo del grafico.
    :param ticks_rotation:  rotacion sobre  x-ticks (etiquetas).
    :param topn: top n de clases a ser graficadas en el barplot.
    :param ax: objetos de eje de matplotlib considerado para la grafica.
    """
    uniq_values = df[attribute].value_counts().head(topn).index
    df = df[df[attribute].isin(uniq_values)]
    data = df.groupby(groupby)[attribute].value_counts(normalize=True).rename('porcentaje').mul(100).reset_index()
    sns.barplot(data = data , x = attribute, y ='porcentaje', hue=groupby,ax=ax)
    plt.xticks(rotation=ticks_rotation)
    plt.title(title)

def boxplot_per_classes(df, attribute, groupby, title=None, ticks_rotation=0, ax=None):
    """
    Grafica el boxplot de atributos para cada clase.
    
    :param df: dataframe  con la data que  sera graficada en el boxplot.
    :param attribute: nombre del atributo a ser graficado en el boxplot.
    :param groupby: nombre del atributo con la clase predictora.
    :param title: titulo del grafico.
    :param ticks_rotation:  rotacion sobre  x-ticks (etiquetas).
    :param ax: objetos de eje de matplotlib considerado para la grafica.
    """
    sns.boxplot(x=groupby, y=attribute, data=df, ax=ax)
    plt.title(title)
    plt.xticks(rotation=ticks_rotation)    

def viz_top_n(df, vargroupby,top_n, title_agg ):
    """
    Grafica el plot N de atributos para cada clase.
    
    :param df: dataframe  con la data que  sera graficada en el boxplot.
    :param vargroupby: nombre del atributo a ser graficado en el boxplot.
    :param top_n: numero de  clases  top N a  graficar.
    :param title_agg: titulo del grafico.
    
    """

    # Generar agregado para  consolidar data
    top_N = df.groupby(vargroupby).agg(
    Total_importe=('Revenue', 'sum'),
    Total_cantidad=('Quantity', 'sum')
    ).reset_index()

    top_productos_ordenado = top_N.sort_values(by='Total_importe', ascending=False)
    # identificar el top  principal  por  cantidad y valor  facturado total
    n = top_n
    top_filtro = top_productos_ordenado.head(n)
    # graficas 
    fig, axes = plt.subplots(ncols= 1, nrows=2 , figsize = (16, 12))

    # grafico de  top n    con mas  importe financiero  registrado
    sns.barplot(x='Total_importe', y=vargroupby, data=top_filtro, ax=axes[0], palette='viridis', hue=vargroupby, legend=False)
    axes[0].set_title(f'Top {n} de {title_agg} con mayor importe financiero')
    axes[0].set_xlabel('importe monetario total')
    axes[0].set_ylabel(title_agg)
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)

    # grafico de top 5 de productos más vendidos
    top_productos_cantidad = top_N.head(n).sort_values(by='Total_cantidad', ascending=False)
    sns.barplot(x='Total_cantidad', y=vargroupby, data=top_filtro, ax=axes[1], palette='viridis', hue=vargroupby, legend=False)
    axes[1].set_title(f'Top {n} de {title_agg} con mayor venta')
    axes[1].set_xlabel('cantidad total')
    axes[1].set_ylabel(title_agg)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()      
