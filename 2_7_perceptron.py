# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 06:23:00 2020

@author: niili

2_7 обчучение перцептрона

Реализуйте метод vectorized_forward_pass класса Perceptron.
n — количество примеров, m — количество входов. Размерность входных данных 
input_matrix — (n, m), размерность вектора весов — (m, 1), смещение (bias) — 
отдельно. vectorized_forward_pass должен возвращать массив формы (n, 1), 
состоящий либо из 0 и 1, либо из True и False.

Обратите внимание, необходимо векторизованное решение, то есть без циклов и 
операторов ветвления.

Первое задание из 
https://nbviewer.jupyter.org/github/stacymiller/stepic_neural_networks_public/blob/master/HW_1/Hw_1_student_version.ipynb
"""


import numpy as np

class Perceptron:

    def __init__(self, w, b):
        """
        Инициализируем наш объект - перцептрон.
        w - вектор весов размера (m, 1), где m - количество переменных
        b - число
        """
        
        self.w = w
        self.b = b

    def forward_pass(self, single_input):
        """
        Метод рассчитывает ответ перцептрона при предъявлении одного примера
        single_input - вектор примера размера (m, 1).
        Метод возвращает число (0 или 1) или boolean (True/False)
        """
        
        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b
        
        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):
        """
        Метод рассчитывает ответ перцептрона при предъявлении набора примеров
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных
        Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
        (элементы вектора - boolean или целые числа (0 или 1))
        """
        
        ## Этот метод необходимо реализовать
        #pass
        # result
        inp_shape = np.shape(input_matrix)
        result = np.zeros((inp_shape[0], 1))
        
        
        # culc result
        for i in range(0, inp_shape[0]):
            # sum function
            res_i = np.dot(input_matrix[[i][:]], self.w) + self.b
            
            # activate function
            if res_i > 0:
                result[[i],[0]] = 1
            else:
                result[[i],[0]] = 0
        
        return result
        
        
    
    def train_on_single_example(self, example, y):
        """
        принимает вектор активации входов example формы (m, 1) 
        и правильный ответ для него (число 0 или 1 или boolean),
        обновляет значения весов перцептрона в соответствии с этим примером
        и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
        (на её основании мы потом построим интересный график)
        """

        ## Этот метод необходимо реализовать
        # culc perceptron answer
        if (np.dot(np.transpose(example), self.w) + self.b) > 0:
            predict = 1
        else:
            predict = 0
        
        # correct the weights
        if predict != y:
            if predict == 1:
                self.w = self.w - example
                self.b = self.b - 1
            else:
                self.w = self.w + example
                self.b = self.b + 1
        # return the errror
        return abs(predict - y)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------   

# for testing
#------------------------------------------------------------------------------
#

# def vectorized_forward_pass(input_matrix):
#         """
#         Метод рассчитывает ответ перцептрона при предъявлении набора примеров
#         input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
#         n - количество примеров, m - количество переменных
#         Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
#         (элементы вектора - boolean или целые числа (0 или 1))
#         """
#         w = np.ones((3,1))
#         b = 0
#         ## Этот метод необходимо реализовать
#         # result
#         inp_shape = np.shape(input_matrix)
#         result = np.zeros((inp_shape[0], 1))
        
        
#         # culc result
#         for i in range(0, inp_shape[0]):
#             # sum function
#             res_i = np.dot(input_matrix[[i][:]], w) + b
            
#             # activate function
#             if res_i > 0:
#                 result[[i],[0]] = 1
#             else:
#                 result[[i],[0]] = 0
        
#         return result
        
        
# x = np.array([[1, 2, 3]])        
        
# print(vectorized_forward_pass(x))

#------------------------------------------------------------------------------
def train_on_single_example(example, y):
    """
    принимает вектор активации входов example формы (m, 1) 
    и правильный ответ для него (число 0 или 1 или boolean),
    обновляет значения весов перцептрона в соответствии с этим примером
    и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
    (на её основании мы потом построим интересный график)
    """
    
    
    w = np.ones((3,1))
    b = 0
    ## Этот метод необходимо реализовать
    if (np.dot(np.transpose(example), w) + b) > 0:
        predict = 1
    else:
        predict = 0
        
    if predict != y:
        if predict == 1:
            w = w - example
            b = b - 1
        else:
            w = w + example
            b = b + 1
    return abs(predict - y)