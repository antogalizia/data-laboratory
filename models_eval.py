# -*- coding: utf-8 -*-
"""
Grupo Pandas
Integrantes: Antonella Galizia, Marco Salcedo
Laboratorio de Datos - Selección y evaluación de modelos de clasificación.

El contenido de este archivo respecta al inciso 2 del trabajo práctico. En el mismo se exploraran modelos de KNN
y se evaluará la performance de cada uno según la cantidad de atributos y valores de k.

"""

# Importaciones
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np

# Lectura del dataset
ruta = r"C:\Users\galiz\Downloads"
data = pd.read_csv(ruta+"\emnist_letters_tp.csv", header=None)


#%%

# Función para rotar las imágenes.
def flip_rotate(image):
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Función para renombrar las columnas del dataframe indicando sus coordenadas en la matriz 28 X 28.
def nombrar_columnas(): 
        columnas = ['letra']
        for i in range (1, 785):
             a = ((i-1) // 28) + 1
             b = ((i-1) % 28) + 1   
             coordenadas = f'({a},{b})'
             columnas.append(coordenadas)
        return columnas


# Función para crear scatterplots
def create_scatterplot(atributos):
    # Crear scatterplots
    #atributos1 = ['491', '519', '548']
    plt.figure(figsize=(15, 5))

    # Crear combinaciones de pares de atributos para scatterplots
    pairs = [(atributos[0], atributos[1]), (atributos[0], atributos[2]), (atributos[1], atributos[2])]

    for i, (x_atributo, y_atributo) in enumerate(pairs):
        plt.subplot(1, len(pairs), i+1)
        sns.scatterplot(data=data, x=x_atributo, y=y_atributo, hue='letra', palette='viridis')
        plt.title(f'Scatterplot: {x_atributo} vs {y_atributo}')

    plt.tight_layout()
    plt.show()
    
    
# Función para crear un mapa de calor según la consulta que se realice sobre los datos.
def heatmap_28_28(array_consulta, titulo, label_cbar):
     matriz_heatmap= flip_rotate(array_consulta)
     df_matriz_heatmap = pd.DataFrame(matriz_heatmap, index = [i for i in range(1, 29)],
                  columns = [i for i in range(1, 29)])
     fig, ax = plt.subplots(figsize = (10, 7))

     plt.title(titulo)
     ax = sns.heatmap(df_matriz_heatmap, annot=False, linewidths=1, linecolor='white', 
            cmap= sns.color_palette('magma', as_cmap=True), 
            cbar_kws={'label': label_cbar })
     ax.set_xlabel('n')
     ax.set_ylabel('m', rotation=0)
     
     
# Función para normalizar los nombres de las columnas
def normalize_column_names(names):
    return [name.strip() for name in names]


# Función para entrenar un modelo KNN con distintos K
def entrenar_KNN_distintosK(k, atributos):
     X = data.query("letra == 'A' or letra == 'L'")[atributos]
     Y = data.query("letra == 'A' or letra == 'L'")[['letra']]
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)  
     
     for i in range(1, k+1):
        model = KNeighborsClassifier(n_neighbors = i)
        model.fit(X_train, Y_train.values.ravel()) 
        Y_pred = model.predict(X_test)
        print("Exactitud del modelo, k = ", i, ":", metrics.accuracy_score(Y_test, Y_pred))
        print("F1-score del modelo para 'A':", metrics.f1_score(Y_test, Y_pred, pos_label='A'))
        print("F1-score del modelo para 'L':", metrics.f1_score(Y_test, Y_pred, pos_label='L'), "\n")


def entrenar_KNN(k, atributos):
    X = data.query("letra == 'A' or letra == 'L'")[atributos]
    Y = data.query("letra == 'A' or letra == 'L'")[['letra']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)  
    
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, Y_train.values.ravel()) 
    Y_pred = model.predict(X_test)
    print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
    print("F1-score del modelo para 'A':", metrics.f1_score(Y_test, Y_pred, pos_label='A'))
    print("F1-score del modelo para 'L':", metrics.f1_score(Y_test, Y_pred, pos_label='L'), "\n")
    
    return Y_pred

#%%

# Renombramos las columnas del dataframe indicando sus coordenadas en la matriz 28 X 28.
columnas = nombrar_columnas()
data.columns = columnas

# Normalizamos los nombres de las columnas del dataframe
data.columns = normalize_column_names(data.columns)

# Extraemos el subconjunto de datos a explorar para ajustar con un modelo KNN.
data = data[(data['letra'] == 'L') | (data['letra'] == 'A')]

# Graficamos el numero de muestras por clase 
fig, ax = plt.subplots()
ax = data['letra'].value_counts().plot.bar()
ax.set_title('Frecuencia de Clases')
ax.tick_params(axis='x', labelrotation=0)   

'''
Como se puede observar, las clases presentan el mismo número de muestras (2400) por lo tanto,
las clases se encuentran balanceadas.
'''

#%%

# Crear el subplot
f, s = plt.subplots(1, 2, figsize=(14, 6))
plt.suptitle('Histogramas de los píxeles para las letras L y A', size='large')

# Filtrar los valores cero
data_L = data[data['letra'] == 'L'].drop(columns=['letra']).melt()
data_L = data_L[data_L['value'] != 0]

data_A = data[data['letra'] == 'A'].drop(columns=['letra']).melt()
data_A = data_A[data_A['value'] != 0]

# Crear histogramas para los píxeles de la letra 'L'
sns.histplot(data=data_L, x='value', bins=40, stat='probability', ax=s[0], color='purple')
s[0].set_title('Distribución de píxeles para la letra L')

# Crear histogramas para los píxeles de la letra 'A'
sns.histplot(data=data_A, x='value', bins=40, stat='probability', ax=s[1])
s[1].set_title('Distribución de píxeles para la letra A')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%

matriz_diferencias_de_medianas_L_A = np.abs(np.median(data.query("letra == 'L'").iloc[:,1:], axis=0) - np.median(data.query("letra == 'A'").iloc[:,1:], axis=0))/255
heatmap_28_28(matriz_diferencias_de_medianas_L_A, 'Diferencia de la mediana del valor en cada coordenada (m, n)', 'Diferencias en la mediana de los valores de cada coordenada')


#%%
# Creamos scatterplots para distintos conjuntos de atributos.

atributos1 = ['(17,16)', '(18,16)', '(19,17)']
create_scatterplot(atributos1)


atributos2 = ['(16,20)', '(21,13)', '(19,18)']
create_scatterplot(atributos2)


atributos3 = ['(14,6)', '(19,12)', '(19,16)']
create_scatterplot(atributos3)


atributos4 = ['(14,6)', '(19,17)','(19,12)', '(19,16)']
create_scatterplot(atributos4)


atributos5 = ['(16,20)', '(20,7)', '(21,13)',  '(19,16)', '(19,14)']
create_scatterplot(atributos5)

atributos6 = ['(16,20)', '(20,7)', '(21,13)',  '(19,16)', '(19,14)', '(7,19)']

atributos7 = ['(16,20)', '(20,7)', '(21,13)',  '(19,16)', '(19,14)', '(7,19)', '(19,20)'] # me dio exactitud 0.978.. para k=7 

#%%
# Entrenamos varios modelos KNN con cada conjunto de atributos para k = 8.

k = 8
y_pred1 = entrenar_KNN(k, atributos1)
y_pred2 = entrenar_KNN(k, atributos2)
y_pred3 = entrenar_KNN(k, atributos3)
y_pred4 = entrenar_KNN(k, atributos4)
y_pred5 = entrenar_KNN(k, atributos5)
y_pred6 = entrenar_KNN(k, atributos6)
y_pred7 = entrenar_KNN(k, atributos7)

'''
 Por el rendimiento de estos modelos se puede observar que los modelos entrenados con más
 cantidad de atributos presentan mayor exactitud y a su vez menos tipos de errores, es decir, 
 de las instancias clasificadas como positivas, hay mas instancias que realmente lo son para 
 estos modelos y de las instancias positivas hay mas instancias que fueron clasificadas como 
 positivas.
'''

#%%
# Evaluamos la performance para distintos valores de k.

entrenar_KNN_distintosK(10, atributos2) # k = 10 mejor performance
print('-------------------------------')
entrenar_KNN_distintosK(10, atributos4) # k = 3 mejor performance
print('-------------------------------')
entrenar_KNN_distintosK(10, atributos5) # k = 6 mejor performance
print('-------------------------------')
entrenar_KNN_distintosK(10, atributos6) # k = 7 mejor performance
print('-------------------------------')
entrenar_KNN_distintosK(10, atributos7) # k = 8 mejor performance




