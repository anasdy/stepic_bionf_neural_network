# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:04:13 2020

@author: niili
1_6_10

читайте данные из файла и посчитайте их средние значения.

На вход вашему решению будет подан адрес, по которому расположен csv-файл, из 
которого нужно считать данные. Первая строчка в файле — названия столбцов, 
остальные строки — числовые данные (то есть каждая строка, кроме первой, 
состоит из последовательности вещественных чисел, разделённых запятыми).

Посчитайте и напечатайте вектор из средних значений вдоль столбцов входных 
данных. То есть если файл с входными данными выглядит как

a,b,c,d
1.5,3,4,6
2.5,2,7.5,4
3.5,1,3.5,2

то ответом будет

[ 2.5  2.   5.   4. ]


файл находится по url-адресу
'https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
"""
import numpy as np

# file object
from urllib.request import urlopen
filename = input()
f = urlopen(filename)

# file open and read (look prev step in course)
file_array = np.loadtxt(f, skiprows = 1, delimiter = ",")

print(file_array.mean(axis = 0))
