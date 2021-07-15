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
def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    
    deltas - ndarray of shape (n, n_{l+1})      errors on l+1 layer
    
    sums - ndarray of shape (n, n_l)            the value of sums functions of 
                                                neurons in layer l for sample 
                                                i in the line i
                                                
    weights - ndarray of shape (n_{l+1}, n_l)   matrix of weights from layer l
                                                to layer l+1

    """
    n_l = np.shape(sums)[1]
    n_delt = (n_l, 1)
    deltas_l = np.zeros(n_delt)
    E = np.ones(n_delt)
    n_samples = np.shape(deltas)[0]
    for sample in range(n_samples):
        activation = np.transpose(sigmoid(sums[sample]))
        mul_1 = np.dot(np.transpose(weights), np.transpose(deltas[sample]))
        mul_2 = mul_1 * activation
        deltas_l_i = mul_2 * (E - activation)
        deltas_l += deltas_l_i
    deltas_l = deltas_l / n_samples
    return deltas_l