from math import exp
from machine_algorithm import gradient_descent
import random

class activation_():
    
    def __init__(self, y_i):
        self.y = y_i
    
    def tanh(self):
        return (exp(self.y)-exp(-self.y))/(exp(self.y)+exp(-self.y))

    def relu(self):
        if self.y < 0:
            return 0
        else:
            return self.y

    def sig(self):
        return 1.0/1.0 + exp(-self.y)


def perceptron(x_i, w_i, bias=1.0):
    y_ = 0.0
    for j in range(len(x_i)):
        y_ += w_i[j]*x_i[j]
    Y = y_ + bias
    return activation_(Y).tanh()


def neural_gradient(x_i, y_i, y_i_p, bias=1.0, alpha=1.0):
    loss = []
    w_i = []
    w_bias = 0.0
    for i in range(len(x_i)):
        loss.append(float(y_i[i]-y_i_p[i])**2)
        w_i.append(loss[i]*x_i[i]*alpha)
        w_bias += loss[i]*bias*alpha
    return w_i, loss, w_bias


############################################## END OF THE PROGRAM ###############################################

test = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], 
        [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], 
        [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1]]
sample = [6.9, 3.1, 4.9, 1.5]
y = [random.random(), random.random(), random.random(), random.random()]
print(gradient_descent(test[1], y[1], 0.0, 0.0))
print(perceptron(test[0], [0.0, 0.0, 0.0, 0.0]))
w_i = neural_gradient(test[0], sample, test[1])[0]

print(neural_gradient(test[0], sample, test[1]))
print(perceptron(test[1], w_i))