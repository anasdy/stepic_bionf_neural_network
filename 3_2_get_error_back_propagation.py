# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:50:04 2021

Функция определения ошибка на слое l через ошибку слоя l+1 для n примеров.

Требуется вернуть вектор deltas_l — ndarray формы (n_l​, 1); 

Все нейроны в сети — сигмоидальные. Функции sigmoid и sigmoid_prime уже 
определены.

@author: niili
"""
import numpy as np



#-----------------------------------------------------------------------------
""" from materials to course. From file HW_1_student_version

"""
## Определим разные полезные функции

def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))
#-----------------------------------------------------------------------------




def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    
    deltas - ndarray of shape (n, n_{l+1})      errors in l+1 layer
    
    sums - ndarray of shape (n, n_l)            the value of sums functions of 
                                                neurons in layer l for sample 
                                                i in the line i
                                                
    weights - ndarray of shape (n_{l+1}, n_l)   matrix of weights from layer l
                                                to layer l+1


    deltas_l = np.dot((weights)^T (deltas)^T) * activation * (1 - activation)
    where activation = (sigmoid(sums))^T
    """
    n_l = np.shape(sums)[1]
    deltas_l = np.zeros(n_l)
    I = np.ones(n_l)
    n_samples = np.shape(deltas)[0]
    for sample in range(n_samples):
        activation = np.transpose(sigmoid(sums[sample]))
        mul_1 = np.dot(np.transpose(weights), np.transpose(deltas[sample]))
        mul_2 = mul_1 * activation
        deltas_l_i = mul_2 * (I - activation)
        deltas_l += deltas_l_i
    deltas_l = deltas_l / n_samples
    return deltas_l

#-----------------------------------------------------------------------------

# to tests

deltas = np.array([[1, 2, 3], [4, 5, 6]])
sums = np.array([[2, 3, 4, 1], [5, 6, 7, 8]])
weights = np.array([[3, 4, 1, 2], [5, 8, 7, 6], [9, 10, 11, 12]])

error = get_error(deltas,sums, weights)

