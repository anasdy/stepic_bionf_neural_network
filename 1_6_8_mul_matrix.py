# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:57:58 2020

@author: niili

1_6_8

На вход программе подаются две матрицы, каждая в следующем формате: на первой 
строке два целых положительных числа n n n и m m m, разделенных пробелом - 
размерность матрицы. В следующей строке находится n⋅ m n \cdot m n⋅ m целых 
чисел, разделенных пробелами - элементы матрицы. Подразумевается, что матрица 
заполняется построчно, то есть первые m m m чисел - первый ряд матрицы, числа
от m+1 до 2⋅m - второй, и т.д.

Напечатайте произведение матриц XY^T, если они имеют подходящую форму,
или строку "matrix shapes do not match", если формы матриц не совпадают
должным образом.


В этот раз мы проделали за вас подготовительную работу по считыванию 
матриц (когда вы начнёте решать, этот код будет уже написан):

x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)

"""

# read X-matrix
x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)

# read Y-matrix
y_shape = tuple(map(int, input().split()))
Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)

# transponse Y matrix
Y = np.transpose(Y)

try:
    mul = np.dot(X,Y)
except ValueError:
    print('matrix shapes do not match')
else:
    print(mul)