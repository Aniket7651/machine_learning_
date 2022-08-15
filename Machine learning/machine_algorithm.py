"""
Created on Aug 6 07:16:32 2022
This file have different type of machine learning algorithms
@author: ANIKET YADAV
"""

from math import exp
import _DATA_processing as ln
from find_Distances_ import euclidean

class pre_processing():
    '''pre processing class is use for feature engineering creating the features and labels, 
    the problem of converting raw data to a dataset which can be
    read by machine is called feature engineering.'''

    def __init__(self, x_set):
        self.x = x_set

    def standardization(self):
        standardize = []
        mean = ln.statistics(self.x).mean()
        dev = ln.statistics(self.x).standard_dev()
        for j in self.x:
            standardize.append((j - mean)/dev)
        return standardize

    def normalization(self):
        normalize = []
        (min_j, max_j) = (min(self.x), max(self.x))
        for j in self.x:
            normalize.append((j - min_j)/(max_j - min_j))
        return normalize

    def imputation(self):
        return (1/float(len(self.x)))*sum(self.x)


def gradient_descent(x, y, w, b, alpha=0.0001):
    w_d = 0.0
    b_d = 0.0
    N = len(x)
    for i in range(N):
        w_d += -2*x[i]*(y[i]-(w*x[i]+b))
        b_d += -2*(y[i]-(w*x[i]+b))

    w = w - (1/float(N))*w_d*alpha
    b = b - (1/float(N))*b_d*alpha
    return w, b
    

def train_model(x, y, w, b, epoch, check_loss, alpha=0.0001):
    for i in range(epoch):
        w, b = gradient_descent(x, y, w, b, alpha)
        if i % check_loss == 0:
            print(f'l2 loss on epoch {i}: {ln.loss_function(y, x, w, b).L2_loss()}')
    return w, b


def linear_prediction(x_i, w_i, b):
    return w_i*x_i + b


def logistic_prediction(x_i, w_i, b):
    p = 1.0 + exp(-w_i*x_i + b)
    return 1.0/p


def gaussian_kernal(z, bandwidth=1.0):
    e = exp((-z)**2/(2*bandwidth)**2)
    return (1/(bandwidth*2*3.14)**0.5)*e


def krenel_gradient(x, X, b, bw=1.0):
    N = len(x)
    w_i = []
    sum_of_all_gaussian = []
    for i in range(N):
        sum_of_all_gaussian.append(gaussian_kernal((x[i]-X[i])/b, bw))
    for i in range(N):
        w_i.append(N*gaussian_kernal((x[i]-X[i])/b, bw)/sum_of_all_gaussian[i])
    return w_i


def kernal_regression(w_i, Y):
    return sum(w_i*Y)/len(Y)


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


def perceptron(x_i, w_i, bias=1):
    y_ = 0.0
    for j in range(len(x_i)):
        y_ += w_i[j]*x_i[j]
    Y = y_ + bias
    return activation_(Y).sig()


def neural_gradient(x_i, y_i, y_i_p, bias=1.0, alpha=1.0):
    error = []
    w_i = []
    w_bias = 0.0
    for i in range(len(x_i)):
        error.append(float(y_i[i]-y_i_p[i]))
        w_i.append(error[i]*x_i[i]*alpha)
        w_bias += error[i]*bias*alpha
    return w_i, w_bias, error
    


############################################ END OF THE PROGRAM #################################################

x_i = [12.3, 23.3, 34.4, 45.6, 66.67, 56.7, 76.5]
y_i = [21.4, 20.0, 23.4, 45.5, 56.6, 34.4, 45.65]
a = [0,1,1,0,1,1,1,0,0,1]
p = [0,0,0,1,0,1,1,0,1,0]
p1 = [0,1,0,1,1,1,1,0,0,0]

# w, b = train_model(x_i, y_i, 0.0, 0.0, 7, 1)
print(gaussian_kernal(25.344555))
# print(activation_(0.21020000).tanh())
# print(perceptron(a, p))
# print(neural_gradient(a, p1, p))
