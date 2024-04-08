import numpy as np

# Con numpy

# empleado_01 = np.array([[20222333, 45, 2, 20000], [33456234, 40, 0, 25000], [45432345, 41, 1, 10000]])
def superan_salario(matriz, umbral):
    res = np.empty((0, 4), dtype=int)
    for i in range(len(matriz)):
        if matriz[i][3] > umbral:
            res = np.vstack([res, matriz[i].astype(int)])
    return res


# output = superan_salario(empleado_01, 15000)
# print(output)

# Agrego dos filas nuevas a la matriz
nueva_f1 = np.array([43967304, 37, 0, 12000])
nueva_f2 = np.array([42236276, 36, 0, 18000])
# empleado_02 = np.vstack([empleado_01, nueva_f1, nueva_f2])

# output = superan_salario(empleado_02, 15000)
# print(output)


# Sin numpy

empleado_01 = [[20222333, 45, 2, 20000], [33456234, 40, 0, 25000], [45432345, 41, 1, 10000]]
empleado_02 = [[20222333, 45, 2, 20000], [33456234, 40, 0, 25000], [45432345, 41, 1, 10000], [43967304, 37, 0, 12000], [42236276, 36, 0, 18000]]
def superan_salario_act1(matriz, umbral):
    res = []
    for i in range(len(matriz)):
        if matriz[i][3] > umbral:
            res.append(matriz[i])
    return res


'''
output = superan_salario_act1(empleado_01, 15000)
output_2 = superan_salario_act1(empleado_02, 15000)
print(output)
print(output_2)
'''


# Si se modifica el orden de las columnas, la funcion no va a devolver el resultado correctamente ya que
# toma la columna de indice 3 para validar la condicion del umbral.

empleado_03 = [[20222333, 20000, 45, 2], [33456234, 25000, 40, 0], [45432345, 10000, 41, 1], [43967304, 12000, 37, 0], [42236276, 18000, 36, 0]]
def superan_salario_act3(matriz, umbral):
    res = []

    for i in range(len(matriz)):
        if matriz[i][1] > umbral:
            salario = matriz[i][1]
            matriz[i][1] = matriz[i][2]
            matriz[i][2] = matriz[i][3]
            matriz[i][3] = salario
            res.append(matriz[i])
    return res


output = superan_salario_act3(empleado_03, 15000)
print(output)