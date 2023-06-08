"""
Created on Aug 14 01:47:32 2022
Many types of distance finding algorithms between two data points, for machine learning
@author: ANIKET YADAV
"""

def cosine(x_j, X):
    d = 0.0
    (a, b) = (0.0, 0.0)
    for j in range(len(X)):
        d += x_j[j]*X[j]
        a += x_j[j]**2
        b += X[j]**2
    return d/(a**0.5)*(b**0.5)


def euclidean(x_j, X):
    euclid = 0.0
    for j in range(len(X)):
        euclid += (X[j]-x_j[j])**2
    return (euclid)**0.5


def euclidean_for_each(p_i, q_i):
    euclidean = (float(q_i)-float(p_i))**2
    return (euclidean)**0.5


def manhatten(x_i, X):
    man = 0.0
    for i in range(len(X)):
        man += (abs(x_i-X[i]))
    return man


def manhatten_for_each(x_i, X_):
    return abs(x_i-X_)

############################################### END OF THE PROGRAM #################################################
test = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], 
        [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], 
        [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1]]
sample =  [0.5, 2.3, 1.3, 3.4]
# for i in test:
#     print(cosine(i,sample))
# for p, q in zip([12.0, 37.3, 21.0, 31.9], [21.4, 20.0, 23.4, 45.5]):
#     print(euclidean_for_each(p, q))
# print(manhatten(5, [2, 3, 6, 7, 5]))