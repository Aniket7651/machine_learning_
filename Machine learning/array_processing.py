"""
Created on Jan 15 03:02:34 2023
Another type of data preprocessing file which is only contains the program that's solve array operation
@author: ANIKET YADAV
"""

def concatenate(met_a, met_b, axis):
    e = []
    if axis == 1:
        for i, j in zip(met_a, met_b):
            e.append(i+j)
        return e
    elif axis == 0:
        return met_a+met_b

def shape(metrix):
    (cols, rows) = (0, 0)
    for i in metrix:
        rows+=1
    for j in metrix[0]:
        cols+=1
    return (rows, cols)

def ones(shape, multiply=1.0, type=float, metrixShape=1):
    metrix = []
    for r in range(shape[0]):
        cols = []
        for c in range(shape[1]):
            if type == int:
                cols.append(int(1.*multiply))
            else: cols.append(1.*multiply)
        metrix.append(cols)
    return metrix*metrixShape

def ones1D(num, multiply=1.0, type=float):
    met = []
    for i in range(num):
        if type == int:
            met.append(int(1.*multiply))
        else: met.append(1.*multiply)
    return met

def dot1D_2D(A, B):
    return [sum(i*j for i,j in zip(row, B)) for row in A]


def transform(mtrix):
    transf = []
    for i in range(len(mtrix[0])):
        lisr = []
        for j in range(len(mtrix)):
            lisr.append(mtrix[j][i])
        transf.append(lisr)
    return transf

def addition(A, B):
    A_B = []
    for i in range(len(A)):
        AB = []
        for a, b in zip(A[i], B[i]):
            AB.append(a+b)
        A_B.append(AB)
    return A_B

def subtract(A, B):
    A_B = []
    for i in range(len(A)):
        AB = []
        for a, b in zip(A[i], B[i]):
            AB.append(a-b)
        A_B.append(AB)
    return A_B
    
def add_2D_1D(A, B):
    A_B = []
    for a in A:
        ab = []
        for i in range(len(a)):
            ab.append(a[i]+B[0][i])
        A_B.append(ab)
    return A_B

def subt_2D_1D(A, B):
    A_B = []
    for a in A:
        ab = []
        for i in range(len(a)):
            ab.append(a[i]-B[0][i])
        A_B.append(ab)
    return A_B

def determinant_2x2(matrix):
    return matrix[0][0]*matrix[1][1]-matrix[1][0]*matrix[0][1]


#################################### END OF THE PROGRAMS ######################################

met = [['a','b','c','d', 2, 4],
       ['e','f','g','h', 3, 3],
       ['i','j','k','l', 4, 2],
       ['m','n','o','p', 5, 1],
       ['q','r','s','t', 6, 0]]

print(transform(met))