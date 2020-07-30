# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:19:44 2020

@author: niili

Найдите оптимальные коэффициенты для построения линейной регрессии.

На вход вашему решению будет подано название csv-файла, из которого нужно 
считать данные.

Задача — подсчитать вектор коэффициентов линейной регрессии для предсказания 
первой переменной (первого столбца данных) по всем остальным. 

Напечатайте коэффициенты линейной регрессии, начиная с β0​, 
через пробел. Проверяется совпадения с точностью до 4 знаков после запятой.

"""
import urllib
from urllib import request

import numpy as np


#------------------------------------------------------------------------------
# read file name
# wait url ro csv-file
fname = input()
f = urllib.request.urlopen(fname)
#------------------------------------------------------------------------------

# read data from file
# we wait csv-file, so delimiter is ','
# and we have a name of cols, so we skip the first row
data = np.loadtxt(f, delimiter = ',', skiprows = 1)    # in the 0-cols we have 
# y-vector, in the other - x-matrix without firs cols with ones

#------------------------------------------------------------------------------
# read y-vector and correct x-matrix
x = data
# read shape of x
x_shape = x.shape
# in y-vector number of row equal the number of row in x-matrix
y = np.zeros((x_shape[0], 1))

for i in range(x_shape[0]):
    y[i] = x[i][0]
    x[i][0] = 1
    i += 1
#------------------------------------------------------------------------------ 
# linear regession

xt = np.transpose(x)
xt_mul_x = np.dot(xt, x)
inv_xt_mul_x = np.linalg.inv(xt_mul_x)
inv_mul_xt = np.dot(inv_xt_mul_x, xt)
b = np.dot(inv_mul_xt, y)

#------------------------------------------------------------------------------ 
#print b
for el in b:
    print(el[0], end = ' ' )