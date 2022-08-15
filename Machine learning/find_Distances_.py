"""
Created on Aug 14 01:47:32 2022
Many types of distance finding algorithms between two data points, for machine learning
@author: ANIKET YADAV
"""

def euclidean(x_i, X):
    euclid = 0.0
    for i in range(len(X)):
        euclid += (X[i]-x_i)**2
    return (euclid)**0.5


def euclidean_for_each(p_i, q_i):
    euclidean = (q_i-p_i)**2
    return (euclidean)**0.5


def manhatten(x_i, X):
    man = 0.0
    for i in range(len(X)):
        man += (abs(x_i-X[i]))
    return man

############################################### END OF THE PROGRAM #################################################
# print(euclidean_for_each(5, 2))
print(manhatten(5, [2, 3, 6, 7, 5]))