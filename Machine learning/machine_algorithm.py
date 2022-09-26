"""
Created on Aug 6 07:16:32 2022
This file have different type of machine learning algorithms
@author: ANIKET YADAV
"""

from math import exp, log2
import _DATA_processing as ln
from find_Distances_ import euclidean, manhatten, cosine

class pre_processing():
    '''pre processing class is use for feature engineering creating the features and labels, 
    the problem of converting raw data to a dataset which can be
    read by machine is called feature engineering.'''

    def __init__(self, x_i):
        self.x = x_i

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
    

class kNN():

    def __init__(self, x_i, X):
        self.x_i = x_i
        self.x = X

    def _distance(self, distance):
        distance = []
        if distance == 'manhatten':
            for i in self.x_i:
                distance.append(manhatten(i, self.x))
        elif distance == 'cosine':
            for i in self.x_i:
                distance.append(cosine(i, self.x))
        else:
            for i in self.x_i:
                distance.append(euclidean(i, self.x))
        
        return sorted(distance), distance

    def nearest(self, y_i, k, distance='euclidean'):
        sorted_list, distance_list = self._distance(distance)
        (lable_, unique, unique_count) = ([], [], [])
        for i in range(k):
            _index_ = distance_list.index(sorted_list[i])
            lable_.append(y_i[_index_])
        
        for i in lable_:
            if i not in unique:
                unique.append(i)
            else:
                pass
        for i in unique:
            unique_count.append(lable_.count(i))
        return unique[unique_count.index(max(unique_count))], {k:v for k,v in zip(unique, unique_count)}


class Decision_tree():

    def __init__(self, y):
        self.y = y

    def find_unique(self, x):
        unique = []
        for u in x:
            if u not in unique:
                unique.append(u)
            else:
                pass
        return unique

    def ID3(self, col):
        (unique_count, id3_S) = ([],[])
        S = len(col)
        for i in self.find_unique(col):
            unique_count.append(col.count(i))
        for item in unique_count:
            id3_S.append(item/S)
        return id3_S, self.find_unique(col)

    def entropy(self, attribute):
        entropyy = 0.0
        for i in self.ID3(attribute)[0]:
            entropyy += -i*log2(i)*1/i
        return entropyy

    def lable_split(self, attribute, lable_attrY='Yes', lable_atterN='No'):
        (counter_yes, counter_no) = ([], [])
        (feature_yes, feature_no) = ([], [])
    
        for u in self.find_unique(attribute):
            for i in range(len(attribute)):
                if attribute[i] == u and self.y[i] == lable_attrY:
                    feature_yes.append('%s'%u)
                if attribute[i] == u and self.y[i] == lable_atterN:
                    feature_no.append('%s'%u)
        
            counter_yes.append(feature_yes.count(u))
            counter_no.append(feature_no.count(u))
        return counter_yes, counter_no

    def information_gain(self, attribute, lable_attrY='Yes', lable_atterN='No'):
        try:
            feature_entropy_ID = 0.0
            _attribute_yes = self.lable_split(attribute, lable_attrY, lable_atterN)[0]
            _attribute_no = self.lable_split(attribute, lable_attrY, lable_atterN)[1]
            id3_P = self.ID3(attribute)[0]

            def f_entropy(attr, yes_, no_):
                entropy = []
                unique_count = []
                for i in self.find_unique(attr):
                    unique_count.append(attr.count(i))
                
                for i in range(len(unique_count)):
                    entropy.append(-(yes_[i]/unique_count[i])*log2(yes_[i]/unique_count[i])-no_[i]/unique_count[i]*log2(no_[i]/unique_count[i]))
                return entropy
            
            feature_entopy = f_entropy(attribute, _attribute_yes, _attribute_no)
            for i in range(len(id3_P)):
                feature_entropy_ID += id3_P[i]*feature_entopy[i]
            return self.entropy(self.y)-feature_entropy_ID
        except ValueError:               # math domain error log2(0) = -infinite
                return 'Input having some errors or too sort'


############################################ END OF THE PROGRAM #################################################

test = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], 
        [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], 
        [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1]]
sample = [6.9, 3.1, 4.9, 1.5]
X = ['m','w','w','s','s','s','w','m','s','s','w','s','w','w','m','s']
Y = ['y','y','y','n','y','y','n','y','y','n','n','y','n','y','n','y']
p1 = [0,1,0,1,1,1,1,0,0,0]
# w, b = train_model(x_i, y_i, 0.0, 0.0, 7, 1)
# print(gaussian_kernal(25.344555))
# print(activation_(0.21020000).tanh())
# print(perceptron(a, p))
x_, y_ = ln.load_csv('A:/BIOINFORMAICS/Machine Learning/Machine learning/docs/dicisionDataset.csv').text_csv()
xF, yF = ln.load_csv('A:/BIOINFORMAICS/Machine Learning/Machine learning/docs/iris.csv').featured_dataset()
# print(y_)
print(kNN(xF, sample).nearest(yF, k=55))
(outlook, temp, humidity, wind) = ([], [], [], [])
for i in range(len(x_)):
    (outlook.append(x_[i][0]), temp.append(x_[i][1]), humidity.append(x_[i][2]), wind.append(x_[i][3]))
    
print('outlook: ', Decision_tree(y_).information_gain(outlook))
print('temprature: ', Decision_tree(y_).information_gain(temp))
print('humidity: ', Decision_tree(y_).information_gain(humidity))
print('wind: ', Decision_tree(y_).information_gain(wind))
# print(neural_gradient(a, p1, p))
