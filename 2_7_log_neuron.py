# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:09:23 2020

@author: niili


2_7 логистический нейрон


"""

import numpy as np

#------------------------------------------------------------------------------
# from teachers

# Определим разные полезные функции

def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    
    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        """
        weights - вертикальный вектор весов нейрона формы (m, 1), weights[0][0] - смещение
        activation_function - активационная функция нейрона, сигмоидальная функция по умолчанию
        activation_function_derivative - производная активационной функции нейрона
        """
        
        assert weights.shape[1] == 1, "Incorrect weight shape"
        
        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        
    def forward_pass(self, single_input):
        """
        активационная функция логистического нейрона
        single_input - вектор входов формы (m, 1), 
        первый элемент вектора single_input - единица (если вы хотите учитывать смещение)
        """
        
        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)
    
    def summatory(self, input_matrix):
        """
        Вычисляет результат сумматорной функции для каждого примера из input_matrix. 
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вектор значений сумматорной функции размера (n, 1).
        """
        # Этот метод необходимо реализовать
        inp_shape = np.shape(input_matrix)
        res_vect = np.zeros((inp_shape[0],1))
        
        for i in range(0, inp_shape[0]):
            res_vect[i][0] += float(np.dot(input_matrix[i], self.w))
                        
        return res_vect
    
    def activation(self, summatory_activation):
        """
        Вычисляет для каждого примера результат активационной функции,
        получив на вход вектор значений сумматорной функций
        summatory_activation - вектор размера (n, 1), 
        где summatory_activation[i] - значение суммматорной функции для i-го примера.
        Возвращает вектор размера (n, 1), содержащий в i-й строке 
        значение активационной функции для i-го примера.
        """
        # Этот метод необходимо реализовать
        
        inp_shape = np.shape(summatory_activation)
        result = np.zeros((inp_shape)) 
        for i in range(0,inp_shape[0]):
            result[i][0] = self.activation_function(summatory_activation[i][0])
        return result 
    
    def vectorized_forward_pass(self, input_matrix):
        """
        Векторизованная активационная функция логистического нейрона.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
        (элементы вектора - float)
        """
        return self.activation(self.summatory(input_matrix))
    

    
    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        """
        Внешний цикл алгоритма градиентного спуска.
        X - матрица входных активаций (n, m)
        y - вектор правильных ответов (n, 1)
        
        learning_rate - константа скорости обучения
        batch_size - размер батча, на основании которого 
        рассчитывается градиент и совершается один шаг алгоритма
        
        eps - критерий остановки номер один: если разница между значением целевой функции 
        до и после обновления весов меньше eps - алгоритм останавливается. 
        Вторым вариантом была бы проверка размера градиента, а не изменение функции,
        что будет работать лучше - неочевидно. В заданиях используйте первый подход.
        
        max_steps - критерий остановки номер два: если количество обновлений весов 
        достигло max_steps, то алгоритм останавливается
        
        Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся) 
        и 0, если второй (спуск не достиг минимума за отведённое время).
        """
        
        # Этот метод необходимо реализовать
        
        # create index range [0:n]
        n = np.shape(y)[0]                  ###
        index_range = np.arange(n)
        # shaffle index range
        np.random.shuffle(index_range)
        i = 0
        for step in range(0,max_steps):
            # until we take all line in X or have algorithm stop
            # create batch: take batch_size line from X and y
            if (i + batch_size) < n:
                indx = index_range[i:batch_size]
                i += batch_size
            else:
                indx = index_range[n - batch_size : n]
                i = 0
            # update mini batch
            # and stop condition
            if (self.update_mini_batch(X[indx], y[indx], learning_rate, eps)):
                return 1
        return 0
    
    def update_mini_batch(self, X, y, learning_rate, eps):
        """
        X - матрица размера (batch_size, m)
        y - вектор правильных ответов размера (batch_size, 1)
        learning_rate - константа скорости обучения
        eps - критерий остановки номер один: если разница между значением целевой функции 
        до и после обновления весов меньше eps - алгоритм останавливается. 
        
        Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции) 
        и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1, 
        иначе возвращаем 0.
        """
        # Этот метод необходимо реализовать

        
        # cost function
        J_cost = J_quadratic(self, X, y)
        
        # gradient
        grad_J = compute_grad_analytically(self, X, y)
        
        # update weights
        self.w -= learning_rate * grad_J
        
        # cost function one more time
        J_cost_new = J_quadratic(self, X, y)
        
        err = abs(J_cost - J_cost_new)
        
        if (err < eps):
            return 1
        else:
            return 0

            
        
    
#------------------------------------------------------------------------------
def J_quadratic(neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.
    Всё как в лекции, никаких хитростей.

    neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)
        
    Возвращает значение J (число)
    """
    
    assert y.shape[1] == 1, 'Incorrect y shape'
    
    return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)

def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,
    
    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать 
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.
    
    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """
    
    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'
    
    return (y_hat - y) / len(y)
    
def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """
    Аналитическая производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам
    
    Возвращает вектор размера (m, 1)
    """
    
    # Вычисляем активации
    # z - вектор результатов сумматорной функции нейрона на разных примерах
    
    z = neuron.summatory(X)
    y_hat = neuron.activation(z)

    # Вычисляем нужные нам частные производные
    dy_dyhat = J_prime(y, y_hat)
    dyhat_dz = neuron.activation_function_derivative(z)
    
    # осознайте эту строчку:
    dz_dw = X

    # а главное, эту:
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)
    
    # можно было написать в два этапа. Осознайте, почему получается одно и то же
    # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
    # grad = np.sum(, axis=0)
    
    # Сделаем из горизонтального вектора вертикальный
    grad = grad.T
    
    return grad


#------------------------------------------------------------------------------

def compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции.
    neuron - объект класса Neuron с вертикальным вектором весов w,
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
    y - правильные ответы для тестовой выборки X,
    J - целевая функция, градиент которой мы хотим получить,
    eps - размер $\delta w$ (малого изменения весов).
    """
    
    # эту функцию необходимо реализовать
    
    
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)
    
    for i in range(len(w_0)):
        
        old_wi = neuron.w[i].copy()
        
        # Меняем вес на больший
        neuron.w[i] += eps
        # Считаем знасение целевой функции с положительной прибавкой веса
        add_cost = J(neuron, X, y)
        
        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi
        
        # Меняем вес на меньший
        neuron.w[i] -= eps
        # Считаем знасение целевой функции с положительной прибавкой веса
        sub_cost = J(neuron, X, y)
        
        
        # Вычисляем приближенное значение градиента
        num_grad[i] = (add_cost - sub_cost)/(2*eps)
        
        
        
        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi
            
    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# for test

#------------------------------------------------------------------------------

# def summatory(input_matrix):
#         """
#         Вычисляет результат сумматорной функции для каждого примера из input_matrix. 
#         input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
#         n - количество примеров, m - количество переменных.
#         Возвращает вектор значений сумматорной функции размера (n, 1).
#         """
        
        
#         # Этот метод необходимо реализовать
#         inp_shape = np.shape(input_matrix)
        
        
        
#         w = np.ones((inp_shape[1], 1))
        
        
#         res_vect = np.zeros((inp_shape[0],1))
        
#         for i in range(0, inp_shape[0]):
#             res_vect[i][0] += float(np.dot(input_matrix[i], w))
                        
#         return res_vect
    
# #------------------------------------------------------------------------------
# def activation(summatory_activation):
#     """
#     Вычисляет для каждого примера результат активационной функции,
#     получив на вход вектор значений сумматорной функций
#     summatory_activation - вектор размера (n, 1), 
#     где summatory_activation[i] - значение суммматорной функции для i-го примера.
#     Возвращает вектор размера (n, 1), содержащий в i-й строке 
#     значение активационной функции для i-го примера.
#     """
#     # Этот метод необходимо реализовать
#     inp_shape = np.shape(summatory_activation)
#     result = np.zeros((inp_shape)) 
#     for i in range(0,inp_shape[0]):
#         result[i][0] = sigmoid(summatory_activation[i][0])
#     return result 
    
# x = np.array([[1, 2, 3]])
# print('sum_funct ', summatory(x))
# print('act_funct ', activation(summatory(x)))


#------------------------------------------------------------------------------
np.random.seed(42)
n = 10
m = 5

X = 20 * np.random.sample((n, m)) - 10
y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
w = 2 * np.random.random((m, 1)) - 1

neuron = Neuron(w)
print(neuron.update_mini_batch(X, y, 0.1, 1e-5))
