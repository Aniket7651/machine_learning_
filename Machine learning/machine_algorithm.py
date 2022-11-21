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
    for j in range(N):
        w_d += -2*x[j]*(y[j]-(w*x[j]+b))
        b_d += -2*(y[j]-(w*x[j]+b))

    w = w - (1/float(N))*w_d*alpha
    b = b - (1/float(N))*b_d*alpha
    return w, b


class multivariate_descent():
    
    def h0_x(self, X, b, W):
        h0 = 0.0
        for x, w in zip(X, W):
            h0 += x*w
        return h0+b

    def cost_F(self, h0x, x_j, y_i):
        loss = 0.0
        for j in x_j:
            loss += (h0x-y_i)*j
        return loss

    def GD(self, loss, W, instance, lr):
        theta_w = []
        for j in W:
            theta_w.append((1/float(instance))*j-lr*loss)
        return theta_w

    def gradient_descent(self, X, b, W, Y, lr=0.01):
        WUpdate = []
        for i in range(len(X)):
            h0 = self.h0_x(X[i], b, W)
            cost = self.cost_F(h0, X[i], Y[i])
            print(cost)
            WUpdate.append(self.GD(cost, W, len(X), lr))
        return WUpdate


def train_model(x, y, w, b, epoch, check_loss, alpha=0.0001):
    for i in range(epoch):
        w, b = gradient_descent(x, y, w, b, alpha)
        if i % check_loss == 0:
            print(f'l2 loss on epoch {i}: {ln.loss_function(y, x, w, b).L2_loss()}')
    return w, b


class LinearRegression():

    def linear_prediction(self, x_i, w_i, b):
        return w_i*x_i + b

    def multivariate_prediction(self, X, W, b):
        wx = 0.0
        for i in range(len(X)):
            wx += X[i]*W[i]
        return wx+b

class LogisticRegression():

    def logistic_prediction(self, x_i, w_i, b):
        p = 1.0 + exp(-w_i*x_i + b)
        return 1.0/p

    def multivariate_prediction(self, X, W, b):
        linear_ = LinearRegression().multivariate_prediction(X, W, b)
        p = 1.0 + exp(linear_)
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

    def ID3(self, attribute):
        (unique_count, id3_S) = ([],[])
        S = len(attribute)
        for i in self.find_unique(attribute):
            unique_count.append(attribute.count(i))
        for item in unique_count:
            id3_S.append(item/S)
        return id3_S, self.find_unique(attribute)

    def entropy(self, ID3_yes, ID3_no):
        if ID3_yes and ID3_no == 0:
            return 0.0
        else:
            entropyy = -(ID3_yes*log2(ID3_yes))-(ID3_no*log2(ID3_no))
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

    def Multiple_entropy(self, attribute, lable_Yes='Yes', lable_No='No'):
        S_a = len(attribute)
        entopy = 0.0
        _attribute_yes = self.lable_split(attribute, lable_Yes, lable_No)[0]
        _attribute_no = self.lable_split(attribute, lable_Yes, lable_No)[1]
        T = [y+n for y, n in zip(_attribute_yes, _attribute_no)]
        for i in range(len(_attribute_yes)):
            (y, n) = (_attribute_yes[i]/S_a, _attribute_no[i]/S_a)
            E = self.entropy(y, n)
            entopy += (T[i]/S_a)*E
        return entopy

    def Gain(self, attribute, lable_Yes='Yes', lable_No='No'):
        ID3 = self.ID3(self.y)[0]
        E_S = self.entropy(ID3[0], ID3[1])
        E_Sa = self.Multiple_entropy(attribute, lable_Yes, lable_No)
        return E_S-E_Sa


class Naive_Bayes():

    def __init__(self, Y):
        self.y = Y

    def lable_probability(self):
        Total = len(self.y)
        yes, no = Decision_tree(self.y).lable_split(self.y)
        (probability_yes, probability_no) = (0, 0)
        for i in range(len(yes)):
            probability_yes += yes[i]
            probability_no += no[i]
        return probability_yes/Total, probability_no/Total

    def attribute_probability(self, attribute):
        yes, no = Decision_tree(self.y).lable_split(attribute)
        (yes_count, no_count) = (0, 0)
        for i in range(len(yes)):
            yes_count += yes[i]
            no_count += no[i]
        unique = Decision_tree(self.y).find_unique(attribute)
        (p_Y, p_N) = ([], [])
        for y, n in zip(yes, no):
            (p_Y.append(y/yes_count), p_N.append(n/no_count)) # type: ignore
        return {k:v for k, v in zip(unique, p_Y)}, {k:v for k, v in zip(unique, p_N)}
    
    def Bayes(self, sample, *attributes):
        (p_yes, p_no) = (1.0, 1.0)
        _Yes, _No = self.lable_probability()
        for atter, s in zip(attributes, sample):
            yes, no = self.attribute_probability(atter)
            p_yes *= yes[s]
            p_no *= no[s]
        Yes = (p_yes*_Yes)/((p_yes*_Yes)+(p_no*_No))
        No = 1-Yes
        lable_dict = {Yes: 'Yes', No: 'No'}
        return lable_dict[max(Yes, No)], max(Yes, No)


############################################ END OF THE PROGRAM #################################################

test = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], 
        [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], 
        [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1]]
sample = [6.9, 3.1, 4.9, 1.5]
X = ['m','w','w','s','s','s','w','m','s','s','w','s','w','w','m','s']
Y = ['y','y','y','n','y','y','n','y','y','n','n','y','n','y','n','y']
p1 = [0,1,0,1,1,1,1,0,0,0]
# w, b = train_model(test[0], sample[0], 0.0, 0.0, 4, 1)
# print(gaussian_kernal(25.344555))
# print(activation_(0.21020000).tanh())
# x_, y_ = ln.load_csv('A:/BIOINFORMAICS/Machine Learning/Machine learning/docs/dicisionDataset.csv').text_csv()
# xF, yF = ln.load_csv('A:/BIOINFORMAICS/Machine Learning/Machine learning/docs/iris.csv').featured_dataset()

X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], 
        [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4]]
Y = [0.072636, 3.172632, 4.9232334, 1.517612, 8.287214, 2.0292334]
W = [0.0, 0.0, 0.0, 0.0]     # [random.random(), random.random(), random.random(), random.random()]

print(multivariate_descent().gradient_descent(X, 0.0, W, Y))
# print(kNN(xF, sample).nearest(yF, k=55))
# (outlook, temp, humidity, wind) = ([], [], [], [])
# for i in range(len(x_)):
#     (outlook.append(x_[i][0]), temp.append(x_[i][1]), humidity.append(x_[i][2]), wind.append(x_[i][3]))
# print('outlook:    ', Decision_tree(y_).Gain(outlook))
# print('temprature: ', Decision_tree(y_).Gain(temp))
# print('humidity:   ', Decision_tree(y_).Gain(humidity))
# print('wind:       ', Decision_tree(y_).Gain(wind))
# print(Naive_Bayes(y_).Bayes(['Sunny', 'Hot'], outlook, temp))
# print(Decision_tree(y_).lable_split(x))
