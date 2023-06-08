"""
Created on Nov 13 08:36:49 2022
These file have seperate deep learning programs like artificial neural networks
and etc other...
@author: ANIKET YADAV
"""

from math import exp
from array_processing import dot1D_2D


def layer_(X, W, Bais, activation_f):   # Bais will be int or float type
    dotProd_XW = dot1D_2D(W, X)
    y_ = []
    y_final = []
    for i in dotProd_XW:
        y_.append(i+Bais)
    for i in y_:
        if activation_f == 'sigmoid':
                y_final.append(1/(1 + exp(-i)))
        elif activation_f == 'relu':
            if i < 0:
                y_final.append(0)
            else:
                y_final.append(i)
        else:
            # tanh function
            y_final.append((exp(i)-exp(-i))/(exp(i)+exp(-i)))
    return y_final


def FeedForwardNN(x, W, bais, activ_func='sigmoid'):
    lay = [layer_(x, W[0], bais[0], activ_func)]
    for i in range(1, len(W)):
        lay.append(layer_(lay[-1], W[i], bais[i], activ_func))
    return lay


class BackPropagation:
    
    def __init__(self, Target, Output, Weight, H, alpha=0.5):
        (self.target, self.output, self.W, self.h, self.alpha) = (Target, Output, Weight, H, alpha)

    def BackPropagation_OutNodes(self):
        err_total = 0.0
        for i in range(len(self.target)):
            err_total += 0.5*(self.target[i]-self.output[i])**2
        err_w = []
        for i in range(len(self.target)):
            dels_err_y1 = -(self.target[i]-self.output[i])
            dels_yFinal = self.output[i]*(1-self.output[i])
            y1_w = self.h[i]
            print(self.h[i])
            err_w.append(dels_err_y1*dels_yFinal*y1_w)

        N = len(W[-1])
        update_w = []
        for i in range(N):
            w_new = []
            for j in range(N):
                w_new.append(W[-1][i][j]-self.alpha*err_w[i])
            update_w.append(w_new)
        return update_w

    def BackPropagation_hiddenNodes(self, x, CurrentLayer, NextLayer):
        dels_totale_h = 0.0
        dels_err = []
        for i in range(len(self.target)):
            dels_yFinal = self.output[i]*(1-self.output[i])
            dels_e_y = 2.0*0.5*(self.target[i]-self.output[i])*(-1)*dels_yFinal
            dels_totale_h += dels_e_y*W[NextLayer][i][0]
        for i in range(len(self.h)):
            e_h = self.h[i]*(1-self.h[i])
            dels_err.append(dels_totale_h*e_h*x[i])

        N = len(W)
        update_w = []
        for i in range(N):
            w_new = []
            for j in range(N):
                w_new.append(W[CurrentLayer][i][j]-self.alpha*dels_err[i])
            update_w.append(w_new)
        return update_w


############################################## END OF THE PROGRAM ###############################################

W = [[[0.15, 0.20], [0.25, 0.30]], [[0.40, 0.45], [0.50, 0.55]], [[0.5, 0.4], [0.4, 0.23]]]
x = [0.05, 0.10, 0.3]
b = [0.35, 0.60, 0.72]
h = [0.5932699921071872, 0.596884378259767, 0.82]
o = [0.7513650695523157, 0.7729284653214625, 0.28]
t = [0.01, 0.99, 0.21]
# print(layer_(h, W[1], 0.60, activation_f='sigmoid'))
print(FeedForwardNN(x, W, b))
Back = BackPropagation(t, o, W, h)
print(Back.BackPropagation_OutNodes())
# for i in range(len(W)-1):
print(Back.BackPropagation_hiddenNodes(x, 0, 1))
# print(layer_(x, W[0], Bais=0.35, activation_f='sigmoid'))